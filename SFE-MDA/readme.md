# SFE-MDA

##### Introduction

This project is a multi-step prediction framework for the sintering endpoint designed for a typical sintering process, which is implemented based on the time series data feature extraction module and the reinforcement learning framework.

Due to the relevant confidentiality agreement, the sintering data used in the algorithm is not open source. However, in order for users to use the framework more flexibly, the overall algorithm can be simply divided into two parts: the time series data feature extractor and the MDA deep reinforcement learning algorithm. The simplified algorithm (removing the expert knowledge module) can be used for various time series data multi-step prediction problems.

The time series data feature extractor extracts the low-dimensional features of high-dimensional time series data through the autoencoder structure, and inputs the features as the state into the RL framework, and the output multi-dimensional actions are the required multi-step prediction values.

##### Document

`main.py`

Implement the MDA algorithm based on the trained TSFE and PSFE models, including data processing, state construction, environment modeling, cycle training, result display, etc.

`MDA_Agent.py`

This file is the MDA agent class functions.

`PSFE.py` and `TSFE.py`

State Feature Extractor. For time series data in other fields, PSFE can be used as a general framework to extract features for use in MDA.

`utils.py`

Other commonly used function libraries, including online data reading, time window construction, outlier processing, evaluation index definition, BTP calculation value fitting and other functions. 