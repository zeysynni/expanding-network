import gym
import gym_cartpole_swingup
import numpy as np
import random
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
from LQR import k_opt
env = gym.make("CartPoleSwingUp-v0")
LR = 1e-3
#env = gym.make("Pendulum-v1", g=9.81)
"""
done = False
env.reset()
while not done:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.render()
"""
def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 8, activation='relu')
    #network = fully_connected(network, 128, activation='relu')
    #network = dropout(network, 0.8)

    network = fully_connected(network, 16, activation='relu')
    #network = fully_connected(network, 256, activation='relu')
    #network = dropout(network, 0.8)

    network = fully_connected(network, 8, activation='relu')
    #network = fully_connected(network, 512, activation='relu')
    #network = dropout(network, 0.8)

    #network = fully_connected(network, 256, activation='relu')
    #network = dropout(network, 0.8)

    #network = fully_connected(network, 128, activation='relu')
    #network = dropout(network, 0.8)

    network = fully_connected(network,1, activation='linear')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='mean_square', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    X = np.array([i[0].reshape([5,]) for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = np.array([i[1].reshape([1,]) for i in training_data]).reshape(-1,1)
    if not model:
        model = neural_network_model(input_size = len(X[0]))
    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model

def show():
    scores = []
    choices = []
    for each_game in range(1):
        score = 0
        game_memory = []
        prev_obs = []
        done = False
        env.reset()
        while not done:
            env.render()
            if len(prev_obs)==0:
                action = env.action_space.sample()
            else:
                if reward > 0.914:
                    #theta = np.arccos(prev_obs[2])
                    #prev_obs = np.array([prev_obs[0],prev_obs[1],theta,prev_obs[4]]).reshape([4,1])
                    #action = k_opt.dot(prev_obs)
                    action = model.predict(prev_obs.reshape(-1,len(prev_obs),1))
                else:
                    action = model.predict(prev_obs.reshape(-1,len(prev_obs),1))
            #action = np.sign(action)
            choices.append(action)   
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score+=reward
            #action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            #action = env.action_space.sample()
            #obs, rew, done, info = env.step(action)
        scores.append(score)
#reward for chosen episode
#reward number
tra = 2
def letsgobrandon(model=False):
    R = []
    training_data = []
    while training_data == []:
        print("searching for data")
        for i in range(tra):
            done = False
            pre_obs = []
            #reward for current episode of each time step
            reward = []
            game_memo = []
            #lets go brandon
            env.reset()
            while not done:
                if model and len(pre_obs)!=0:
                    if np.random.uniform(0,1)>0.6:
                        act = model.predict(pre_obs.reshape(-1,len(pre_obs),1))
                    else:
                        act = (np.random.uniform(0,3))*model.predict(pre_obs.reshape(-1,len(pre_obs),1))
                else:
                    act = env.action_space.sample()
                #act = np.sign(act)
                obs, rew, done, info = env.step(act)
                if len(pre_obs) > 0:
                    game_memo.append([pre_obs, act])
                    reward.append(rew)
                pre_obs = obs
                #env.render()
            if R != []:
                #print("R",len(R))
                #print("R",np.sum(max(R)))
                #print("R",sum(max[R]))
                #print("reward",len(reward))
                #print("reward",sum(reward))
                if sum(reward) > np.sum(max(R)):
                    R = []
                    R.append(sum(reward))
                    traning_data = []
                    #for data in game_memo:episode, steps
                    for data in game_memo:
                        training_data.append([data[0],data[1]])
                elif sum(reward) == np.sum(max(R)):
                    R.append(sum(reward))
                    #for data in game_memo:episode, steps
                    for data in game_memo:
                        training_data.append([data[0],data[1]])
            else:
                R.append(sum(reward))
                #for data in game_memo:episode, steps
                for data in game_memo:
                    training_data.append([data[0],data[1]])
            s = rew
            """
            if max(reward) >= s:
                R.append(reward)
                #for data in game_memo:episode, steps
                for data in game_memo:
                    training_data.append([data[0],data[1]])
            """
    #s = max(max(R))
    print("len training data",len(training_data))
    print(len(R))
    return training_data,R,s

#training_data,R = letsgobrandon()
#model = train_model(training_data)
#model.save("hej.model")
#X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
#model = neural_network_model(input_size = len(X[0]))
#neural_network_model(input_size)
#model = tflearn.DNN(network)
#model.load("hej.model")
""
s = 0
model = False
while s <= 4:
    if not model:
        training_data,R,s = letsgobrandon()
        break
        ##model = train_model(training_data)
    else:
        print("!!!!!!",s)
        #pre_obs = training_data[3][0]
        #print(model.predict(pre_obs.reshape(-1,len(pre_obs),1)))
        #print(training_data[3][1])
        #break 
        training_data,R,s = letsgobrandon(model)
        print("len",len(R))
        model = train_model(training_data,model = model)
    #model.save("hej.model")
    ##show()
    #s = s + 1
scores = []
choices = []
""
