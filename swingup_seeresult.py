import gym
import gym_cartpole_swingup
#from mlp_test import mlp
from mlp_expanding import mlp
import numpy as np
import torch
import matplotlib.pyplot as plt
#shows animation at the end of each policy update
env = gym.make("CartPoleSwingUp-v0")
#nn = mlp(0.1,1)
#w1,w2,w3,w4 = nn.weights()
def show(weights,nn,weights_var,nn_var,index):#w1_,w2_,w3_,w4_,
    for each_game in range(1):
        prev_obs = []
        done = False
        env.reset()
        while not done:
            env.render()
            if len(prev_obs)==0:
                action = env.action_space.sample()
            else:
                prev_obs = torch.tensor(new_observation,dtype=torch.float32).reshape([1,5])
                m = nn.forward(prev_obs,weights,index)
                var = nn_var.forward_var(prev_obs,weights_var)
                action = np.array(torch.normal(mean=float(m), std=var.detach()))
            new_observation, reward, done, info = env.step(action)
            #print(done)
            #print(info)
            prev_obs = new_observation

def diagramm(T,m,upper,lower):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for i in range(T):
        plt.scatter(i,m[i],color="r")
        plt.scatter(i,upper[i],color="yellow")
        plt.scatter(i,lower[i],color="yellow")
        plt.plot([i,i],[upper[i],lower[i]],color="yellow")
    m = np.array(m)
    upper = np.array(upper)
    lower = np.array(lower)
    np.save("m.npy",m)
    np.save("upper.npy",upper)
    np.save("lower.npy",lower)
    plt.show()
    return
