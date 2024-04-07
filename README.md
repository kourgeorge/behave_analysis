# Bayesian Modeling of Reward-Dependent Sequential Decision Dynamics
This repository contains the code implementation, environment setup, and scripts used in our study titled: "Bayesian Modeling of Reward-Dependent Sequential Decision Dynamics".

### Context-Aware Hidden Markov Model (CAHMM)
The Context-Aware Hidden Markov Model (CAHMM) is designed to capture the dynamics of an agent's behavior in a dynamic environment by incorporating context information, such as observed states, actions, and rewards. 
This model aims to reveal the underlying policies employed by the agent and how they transition based on rewards and environmental cues.

### Bayesian Interpretable Reward-dependent (BIRD) Inference Method
The Bayesian Interpretable Reward-dependent (BIRD) inference method is utilized in conjunction with CAHMM to estimate the set of policies the agent employs. 
BIRD incorporates observed information and model assumptions to uncover the dynamics of a non-optimal agent in both discrete and continuous state environments.

### T-BIRD and N-BIRD Identification Approaches
1. T-BIRD (Tabular-BIRD): Utilizes a tabular representation to describe the state-action probabilistic relation, suitable for smaller state spaces.
2. N-BIRD (Neural-BIRD): Employs a neural-based state-action mapping, better suited for larger or continuous state spaces.
Both T-BIRD and N-BIRD operate within the Bayesian framework, offering different representations and complexities in estimating the set of policies in the CAHMM model.
