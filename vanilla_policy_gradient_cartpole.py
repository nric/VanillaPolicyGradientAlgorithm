"""
This is an Tensorflow.Keras (TF 2.0) based implementation of a vanilla Policy Gradient learner
to solve OpenAi Gym's Cartpole. 
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

#%%
#----------Imports----------
import gym
import numpy as np
from collections import deque as DQ

import tensorflow as tf
import tensorflow.keras.backend as K

#%%
#------------DEFINE AGNET CLASS AND FUCTIONS--------------
class PG_Agent(object):
    def __init__(self,input_dim,output_dim,hidden_dims=[32,32]):
        self.input_shape = input_dim
        self.input_size = input_dim[0]
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.losses = [] # protocol of the losses
        self.log = DQ(maxlen=10) # protocol for debug
        self.GAMMA = 0.99
        self.LEARNING_RATE = 1e-2
        self.ENTROPY_BETA = 0.01
        self._build_model()
  
        
    def _build_model(self):
        #the main input for the NN
        model_input = tf.keras.layers.Input(shape = self.input_shape,name="input_X")
        #dont get confused with the advantage layer. This is important for the input
        #but has no effect on the hidden layers and the model_output but only on the loss
        advantage = tf.keras.layers.Input(shape=[1],name="advantages")
        #add hidden dims
        model = model_input
        for dim in self.hidden_dims:
            model = tf.keras.layers.Dense(dim,activation='relu')(model)
        #Output layer with actions as neurons and softmax activation to make them probabilities
        model_output = tf.keras.layers.Dense(self.output_dim,activation='softmax')(model)

        #define custom loss function INSIDE this fuction so it has access to advantage layer object
        def custom_loss(y_true,y_pred):
            # L = -Q(s,a) * Log(Policy(s,a)) where Q(s,a) is the advatage(discounted rewards) and 
            # Policy is the the action probabiliy for this specific action only
            # actual: 0 predict: 0 -> log(0 * (0 - 0) + (1 - 0) * (0 + 0)) = -inf
            # actual: 1 predict: 1 -> log(1 * (1 - 1) + (1 - 1) * (1 + 1)) = -inf
            # actual: 1 predict: 0 -> log(1 * (1 - 0) + (1 - 1) * (1 + 0)) = 0
            # actual: 0 predict: 1 -> log(0 * (0 - 1) + (1 - 0) * (0 + 1)) = 0
            lik = y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred)
            log_lik = K.log(lik)
            #mean to apply over all episodes
            loss_policy = K.mean(log_lik * advantage, keepdims=True)
            #Optionally: Entropy loss is: H(probability) = Sum(Probability*LOG(probability))
            loss_entropy = - self.ENTROPY_BETA * K.sum(-(lik * log_lik))
            #calc and return total loss
            loss = loss_policy #+ loss_entropy
            #loss = K.clip(loss,-10,10)
            return loss

        #make a trainable model mich also takes the advantages as input, nedded for the loss fucntion
        self.model_train = tf.keras.Model(inputs=[model_input,advantage],outputs=model_output)
        self.model_train.compile(loss=custom_loss,optimizer=tf.keras.optimizers.Adam(lr=self.LEARNING_RATE))
        #and also make an predict only model which does not has a loss fuction and hence does not need
        #the advantage layer/input
        self.model_predict = tf.keras.Model(inputs=[model_input],outputs=model_output)
    

    def calc_discounted_rewards(self,reward_lst):
        #Q(k,t) = Sigma_i(gamma*reward_i) with t=step and k=episode
        prev_val = 0
        out = []
        for val in reward_lst:
            new_val = val + prev_val * self.GAMMA
            out.append(new_val)
            prev_val = new_val
        #remember to flip
        return np.array(out[::-1])
    

    def fit(self,S,A,R):
        """Makes a training on a given batch.
        Param:
            :S: 1D Numpy Array of States
            :A: 1D Numpy Array of Actions taken at in state
            :R: 1D Numpy Array of Rewards given after action in state
        Return:
            Loss of last training
        """
        #get discounted rewards
        discounted_rewards = R
        #Baseline normalization of the rewards
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        #one-hot the action taken
        actions_train = np.zeros([len(A), self.output_dim])
        actions_train[np.arange(len(A)), A] = 1

        #call keras.model.train_on_batch and protocol loss
        #print(f"S{S.shape}  discounted_rewards{discounted_rewards.shape}  actions_train{actions_train.shape}")#debug
        loss = self.model_train.train_on_batch([S, discounted_rewards], actions_train)
        self.log.append([S,A,actions_train,R,discounted_rewards,loss])
        self.losses.append(loss)
        return loss
    

    def get_action(self,state):
        """Returns an action for a given State.
        The probability for an action is given from the neural net which is fitet to rewards for
        actions at given states from past experiences. 
        Hence, for a action at a given state that is known to 
        give an above average reward, the probability will be high.        
        """
        action_prob = np.squeeze(self.model_predict.predict(state))
        # return np.random.choice(self.output_dim,1,p=action_prob)[0]
        return np.random.choice(range(self.output_dim),p=action_prob)

        

def run_n_episode(env,agent,n,render=False):
    """Multistep Bellman Rollouts
    Plays N episodes, Stores SAR, Calls Training after that.
    Param:
        :env:
        :agent:
        :n:
        :render:
    Returns:
        Average of total reward over the n episodes.
    """
    S,A,R = [],[],[]
    total_reward = 0

    for episode in range(n):
        done = False        
        s = env.reset()
        while not done:
            if render and episode < 5: env.render()
            a = agent.get_action(np.reshape(s,[1,agent.input_size])) #reshape important
            s_next,r,done,_ = env.step(a)
            total_reward += r
            S.append(s)
            A.append(a)
            R.append(r)
            s = s_next
    
    #Call training with the transistions
    S = np.array(S)
    A = np.array(A)
    R = np.array(agent.calc_discounted_rewards(R)) #rewards should be discounted for each episode.
    #print(f"{S.shape} {A.shape} {R.shape}") #debug
    loss = agent.fit(S,A,R)
    print(f"Training on batch complete. Loss: {loss}")
    return total_reward/n



#%%
#----------------main---------------
CYCLES = 1000
EPISODE_BATCH_SIZE = 50
RENDER_EVERY_N = 10 #render every Nth episode


env = gym.make("CartPole-v0")
agent = PG_Agent(env.observation_space.shape, env.action_space.n, hidden_dims=[32, 32])

for episode in range(CYCLES):
    avg_reward = run_n_episode(
        env,
        agent,
        EPISODE_BATCH_SIZE,
        render=True if episode % RENDER_EVERY_N == 0 else False)
    print(f"episode: {episode*EPISODE_BATCH_SIZE}  reward: {avg_reward}")

env.close()
