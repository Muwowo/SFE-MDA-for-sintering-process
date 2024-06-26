import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 64)
		self.l2 = nn.Linear(64, 64)
		self.l3 = nn.Linear(64, action_dim)
		
		self.max_action = max_action
		
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.sigmoid(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		# Q1~Q3构建不同结构，表示裁判的不同喜好
		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 32)
		self.l2 = nn.Linear(32, 32)
		self.l3 = nn.Linear(32, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 64)
		self.l5 = nn.Linear(64, 64)
		self.l6 = nn.Linear(64, 1)

		# Q3 architecture
		self.l7 = nn.Linear(state_dim + action_dim, 128)
		self.l8 = nn.Linear(128, 128)
		self.l9 = nn.Linear(128, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)

		q3 = F.relu(self.l7(sa))
		q3 = F.relu(self.l8(q3))
		q3 = self.l9(q3)
		return q1, q2, q3

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1
	
	def Q2(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l4(sa))
		q1 = F.relu(self.l5(q1))
		q1 = self.l6(q1)
		return q1
	
	def Q3(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l7(sa))
		q1 = F.relu(self.l8(q1))
		q1 = self.l9(q1)
		return q1


class MDA(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		min_action,
		batch_size,
		discount,
		tau,
		policy_noise,
		noise_clip,
		policy_freq,
		target_freq
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)

		self.max_action = max_action
		self.min_action = min_action
		self.batch_size = batch_size
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.target_freq = target_freq

		self.total_it = 0

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (self.actor_target(next_state) + noise).clamp(self.min_action, self.max_action)
 
			# Compute the target Q value
			target_Q1, target_Q2, target_Q3 = self.critic_target(next_state, next_action)
			stacked_target_Q = torch.cat((target_Q1, target_Q2, target_Q3), dim=1)
			stacked_target_Q, _ = torch.topk(stacked_target_Q, 3, dim=1)
			target_Q = stacked_target_Q[:, 1].unsqueeze(1)
			# target_Q = torch.min(target_Q1, target_Q2, target_Q3)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2, current_Q3 = self.critic(state, action)
		Q = (current_Q1 + current_Q2 + current_Q3) / 3
		# Compute critic loss
		
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + F.mse_loss(current_Q3, target_Q)
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -(self.critic.Q1(state, self.actor(state)).mean() + self.critic.Q2(state, self.actor(state)).mean() + self.critic.Q3(state, self.actor(state)).mean()) / 3
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
		if self.total_it % 1000 == 0:
			print('Q:', Q.mean())
			# print(target_Q.size())

		if self.total_it % self.target_freq == 0:
			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		
		return current_Q1.mean().to('cpu').detach().numpy(), current_Q2.mean().to('cpu').detach().numpy(), current_Q3.mean().to('cpu').detach().numpy(), stacked_target_Q[:, 0].unsqueeze(1).mean().to('cpu').detach().numpy(), stacked_target_Q[:, 1].unsqueeze(1).mean().to('cpu').detach().numpy(), stacked_target_Q[:, 2].unsqueeze(1).mean().to('cpu').detach().numpy()
		# return current_Q1.mean().to('cpu').detach().numpy(), current_Q2.mean().to('cpu').detach().numpy(), current_Q3.mean().to('cpu').detach().numpy(), target_Q1.mean().to('cpu').detach().numpy(), target_Q2.mean().to('cpu').detach().numpy(), target_Q3.mean().to('cpu').detach().numpy()
	
	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic", _use_new_zipfile_serialization=False)
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer", _use_new_zipfile_serialization=False)
		
		torch.save(self.actor.state_dict(), filename + "_actor", _use_new_zipfile_serialization=False)
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer", _use_new_zipfile_serialization=False)

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		