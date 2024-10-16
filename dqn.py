# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:12:58 2024

@author: sundari
"""

# importing required libraries
import numpy as np
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import random
from collections import deque
import pickle
from ffrnn import FF

# import wandb
# # initialise wandb
# wandb.init(
#     project = "DQN_with_image_with_LSTM",
#     config = {
#         "learning_rate" : 0.001,
#         "batch_size" : 32,
#         "explore_frames": 1000,
#         "update_target": 5000,
#         "memory_size": 2000,
#         "architecture" : "6_6_3_3" # number of feature maps/nodes in a layer   
#     }
# )

# Defining hyperparameters
seed = 42
gamma = 0.99
epsilon_min = 0.01
epsilon_max = 0.9
epsilon = epsilon_max

num_actions = 3

learning_rate = 0.001
batch_size = 32
num_episodes = 10000 #1000
epoch_num = 10000

memory_buffer_size = 5000 # last 20 episodes
explore_frames = 2500 
num_frames_input = 4
max_steps_per_episode = 100 
update_target = 5000
update_after_actions = 4

seq_len = 1
input_dim = 6
# define the network
class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.fcFeat1 = layers.RNN(FF(12), input_shape=(seq_len, input_dim)) #layers.LSTM(12)#, activation = tf.nn.relu)
        self.fcFeat2 = layers.Dense(12, activation = tf.nn.relu)
        self.fcFeat3 = layers.Dense(6, activation = tf.nn.relu)
        self.out = layers.Dense(num_actions, activation = "linear")    
    def call(self,x):
        feat1 = self.fcFeat1(x)
        feat2 = self.fcFeat2(feat1)
        # a = np.array([0,0,0,0,1,1,1,1,1,1,1,1],dtype='float32')
        # feat2 = feat2 * tf.convert_to_tensor(a)
        feat3 = self.fcFeat3(feat2)
        Q = self.out(feat3)
        return Q
    def chooseAct(self,x):
        Q = self.call(x)
        act = tf.argmax(Q,axis = 1)
        # convert to int to be sent to the environment
        act = int(np.squeeze(act))
        return act

# defining Memory buffer
class Memory(object):
    def __init__(self,memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen = memory_size)
    def add(self,experience) -> None:
        self.buffer.append(experience)
    def size(self):
        return len(self.buffer)
    def sample(self,batch_size: int):
        if batch_size > len(self.buffer): batch_size = len(self.buffer)
        indices = np.random.choice(np.arange(len(self.buffer)), size = batch_size, replace = False)
        return [self.buffer[i] for i in indices]
    def clear(self):
        self.buffer.clear()

# send action to environment
import argparse
from peaceful_pie.unity_comms import UnityComms
import base64
from PIL import Image
import cv2
from io import BytesIO

def sendtoEnv(args: argparse.Namespace, nested_ab: str) -> None:
    unity_comms = UnityComms(port = args.port)
    ab = nested_ab
    unity_comms.HandleMessage(messageNew = ab)

def getfromEnv(args: argparse.Namespace) -> str:
        
    unity_comms = UnityComms(port = args.port)
    reward_env = readb64_string(unity_comms.GetReward())
    int_reward = int.from_bytes(reward_env,"big",signed = "True")
    if int_reward-48 > 1: reward = 1
    else: reward = 0 
    done_signal_env = readb64_string(unity_comms.GetDone())
    int_done = int.from_bytes(done_signal_env,"big",signed = "True")
    done = int_done - 48
    
    ball_x = float(readb64_string(unity_comms.GetballXPos()).decode())
    ball_y = float(readb64_string(unity_comms.GetballYPos()).decode())
    ball_z = float(readb64_string(unity_comms.GetballZPos()).decode())
    playr_x = float(readb64_string(unity_comms.GetplayrXPos()).decode())
    playr_y = float(readb64_string(unity_comms.GetplayrYPos()).decode())
    playr_z = float(readb64_string(unity_comms.GetplayrZPos()).decode())
    state = [ball_x,ball_y,ball_z,playr_x,playr_y,playr_z]
    
    return reward, done, state, int_reward

def readb64_string(b64_s):
    reward_str = base64.b64decode(b64_s)
    return reward_str
    
# Initialise and/or load models
model_name = "position_data_as_input_with_FF"

modelQ = DQN()
# load_modelQ_name = "models/healthy_model_"+model_name+"_ep_"+str(10000)
# modelQ.load_weights(load_modelQ_name)

targetQ = DQN()
# # targetQ.set_weights(modelQ.get_weights())
# load_targetQ_name = "models/target_4_healthy_model_"+model_name+"_ep_"+str(10000)
# targetQ.load_weights(load_targetQ_name)

mse = tf.keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate)
memory_buffer = Memory(memory_buffer_size)
# with open('memory_buffer_'+str(epoch_num - num_episodes)+'_epochs_'+model_name+'.pkl', 'rb') as f:
#     memory_buffer.buffer = pickle.load(f)
parser = argparse.ArgumentParser()
parser.add_argument('--port',type = int,default = 9000)
args = parser.parse_args()
reward_history = []
loss_history = []
loss_history_episode = []
frame_num = 0
for episode in range(num_episodes): #while True:
    episode_reward = 0
    args.port = 9010 # port number for receiving from Unity
    _,_,state,_ = getfromEnv(args) # get information from the game - similar to env.step
    reward_past = 0
    t = 1
    while True:
        frame_num += 1
        prob = random.random()
        if prob < epsilon or frame_num < explore_frames:
            shot = np.random.choice(np.arange(3))
        else:
            state_tensor = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(state),0),0)
            shot = modelQ.chooseAct(state_tensor)#(features_op)
        action_inp = str(random.randint(0, 1))+"_"+str(random.randint(0, 1))+"_"+str(shot)+"_"+str(0)+"_"+str(0)
        args.port = 9000
        sendtoEnv(args,action_inp)
        args.port = 9010
        reward_current,done,next_state,reward_original = getfromEnv(args)
        diff = reward_current - reward_past
        if diff < 0: reward = -20
        else: reward = +1
        # reward = reward_current - reward_past
        reward_past = reward_current
        episode_reward += reward
        epsilon -= (epsilon_min - epsilon_max)/explore_frames
        memory_buffer.add((state,next_state,shot,reward,done))
        if memory_buffer.size() > 128 and frame_num % update_after_actions:
            frame_num += 1
            sample_batch = memory_buffer.sample(batch_size)
            batch_sample_state,batch_sample_next_state,batch_sample_action,batch_sample_reward,batch_sample_done = zip(*sample_batch)
            batch_sample_state = np.asarray(batch_sample_state)
            batch_sample_next_state = np.asarray(batch_sample_next_state)
            batch_sample_action = np.asarray(batch_sample_action)
            batch_sample_reward = np.asarray(batch_sample_reward)
            batch_sample_done = np.asarray(batch_sample_done)
            batch_sample_next_state = tf.expand_dims(batch_sample_next_state, 1)
            q_estimate = targetQ(batch_sample_next_state)
            y = batch_sample_reward + gamma*(tf.reduce_max(q_estimate, axis = 1))
            masks = tf.one_hot(batch_sample_action,num_actions)
            with tf.GradientTape() as tape:
                q_values = modelQ(batch_sample_next_state)
                q_action = tf.reduce_sum(tf.multiply(q_values,masks),axis= 1)
                loss = mse(y,q_action)
                loss_history.append(loss)
            # Backpropagation
            grads = tape.gradient(loss,modelQ.trainable_variables)
            optimizer.apply_gradients(zip(grads,modelQ.trainable_variables))
        
        if frame_num % update_target == 0:
            targetQ.set_weights(modelQ.get_weights())
            template = "Running Reward: {:2f} with loss: {:2f} at epsiode {}, frame count {}"
            print(template.format(episode_reward,np.mean(loss_history),episode,frame_num))
        t += 1
        
        if done == 1 or t > max_steps_per_episode:
            reward_history.append(episode_reward)
            loss_history_episode.append(np.mean(loss_history))
            break
    
modelQ.save_weights("models/healthy_model_"+model_name +"_ep_"+str(epoch_num))
targetQ.save_weights("models/Target_4_healthy_model_"+model_name +"_ep_"+str(epoch_num))
print("Model Saved")

x = np.arange(len(reward_history))
plt.figure()
plt.plot(x,reward_history,color='blue')
plt.xlabel("Number of episodes")
plt.ylabel("Episode Reward")
plt.title("Reward obtained in each episode")

plt.figure()
plt.plot(x,loss_history_episode,color='orange')
plt.xlabel("Number of episodes")
plt.ylabel("Loss obtained")
plt.title("Loss at each episode")

mean_reward_dq = np.zeros((100,1))
mean_reward = np.zeros((int(num_episodes/100)))
j = 0
k = 0
for i in range(len(reward_history)):
    mean_reward_dq[k,0] = reward_history[i]
    k += 1
    if i % 100 == 0:
        mean_reward[j] = np.mean(mean_reward_dq)
        j += 1
        k = 0
        mean_reward_dq = np.zeros((100,1))
    
x = np.arange(len(mean_reward))
plt.figure()
plt.plot(x,mean_reward,color='blue')
plt.xlabel("Number of episodes")
plt.ylabel("Episode Reward")
plt.title("Reward obtained in each episode")
plt.savefig('healthy_reward_avg_'+str(epoch_num)+'_epochs_'+model_name+'.png')

# for i in range(len(reward_history)):
#     wandb.log({"Reward History": reward_history[i],"Loss History": loss_history_episode[i]})

# for i in range(len(mean_reward)):
#     wandb.log({"Average Reward per 100 episode": mean_reward[i]})

# # [optional] finish the wandb run, necessary in notebooks
# wandb.finish()

# Save the variable using pickle
with open('healthy_reward_history_'+str(epoch_num)+'_epochs_'+model_name+'.pkl', 'wb') as f:
    pickle.dump(reward_history, f)
    
m = [*memory_buffer.buffer]
with open('healthy_memory_buffer_'+str(epoch_num)+'_epochs_'+model_name+'.pkl', 'wb') as f:
    pickle.dump(m, f)
    
