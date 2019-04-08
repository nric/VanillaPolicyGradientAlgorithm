# VanillaPolicyGradientAlgorithm
This is an Tensorflow.Keras (TF 2.0) based implementation of a vanilla Policy Gradient learner to solve OpenAi Gym's Cartpole. 

While pretty rudementary, it has some enhancements to make it more effektive:

- Multistep Bellman Rollout: Play several episodes for gathering more quality statistics on SAR

- Baseline Normalized Rewards: Rather than pure rewards use differences to basline.

- Entropy Bonus: Encourage the network to keep some uncertainty to avoid getting stuck
    in a local minima
    
This is part of the course Move37 Chapter 8.2 but rewritten to Tensorflow/Keras. More credits go
to https://github.com/breeko/Simple-Reinforcement-Learning-with-Tensorflow/blob/master/Part%202%20-%20Policy-based%20Agents%20with%20Keras.ipynb

It is written and run in Visual Studio Code using Ipykernel. 

Todo: The Entropy bonus has an issue (gets too large) and is commented out in this version
"""
