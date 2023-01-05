import numpy as np
import torch
from swingup_data import collect_data  #sample episode
from mlp_expanding import mlp
from swingup_seeresult import show  #show simulation
from mlp_v import v_approx  #state value estimation
import warnings
from weighted_lr import lr
warnings.filterwarnings("ignore")


#a code which does the learning of policy
def if_reset(average, average_reward, thr, red=False, res=False):
    global count, monitor, count_1
    #sometimes it stays long at the beginning for some reason
    if average > thr[0] or monitor == 1:
        monitor = 1
        if average < np.max(average_reward):
            count += 1
        else:
            count = 0
        if count > thr[1]:  # or avr < threshold*np.max(avr_reward):
            red = True
            count = 0
            count_1 += 1
        if count_1 > thr[2]:
            res = True
            count_1 = 0
        print("count", count)
        print("monitor", monitor)
    return red, res


seed = 0
tra = 100
tra_size = 50
decay = 0.7  #lr decay
l_rate = .02
l_rate_var = .02
lr_min = 0.005
size = [[5, 1], [1, 1]]
size_var = [[5, 10], [10, 4], [4, 2], [2, 1]]
threshold = [150, 5, 3]  #begin to count/reduce/reset
threshold_node, threshold_layer = 10, 10

start = 1
T = 300

var_decay = 0.995  #var decay
decay_value = 0.97  #decay for calculating value function
index = {}
avr_reward, best, worst, m, upper, lower, avr_var = [[] for x in range(7)]
regression = lr(0.6, 60, 0.5, 0.00001)  #most_import_w_percentage,chosen_data_size,l_for_kernel,ridge_regression
data_collector = collect_data(tra, decay_value, seed)
dc = collect_data(tra_size, decay_value, seed)  #samples used for changing size
nn = mlp(tra, l_rate, seed, size, decay=1.)
nn_var = mlp(tra, l_rate_var, seed, size_var, decay=1.)
torch.manual_seed(seed)
weights = nn.weights()
weights_var = nn_var.weights_var()
monitor, count, count_1 = 0, 0, 0
if start != 1:
    weights, weights_var, size, v, index = data_collector.load_w()
print("w4", weights[-1])
print("w4_var", weights_var[-1])
show(weights, mlp(tra, l_rate, seed, size, decay=0.), weights_var, nn_var, index)

for epo in range(start, T):
    print(epo)
    training_data, R, actions, mean, variance = data_collector.letsgobrandon1(weights, nn, weights_var, nn_var, index)
    avr_reward, best, worst, avr, m, upper, lower = data_collector.save(R, tra, avr_reward, best, worst, m, upper,
                                                                        lower, avr_var, variance)
    print("avr var per step", sum([sum(a)/len(a) for a in variance])/len(variance))
    print("average reward for this epoch", avr)
    print(avr_reward)
    v_func = data_collector.train_data_v(training_data, R)
    if epo == 1:
        v = v_approx(5, 64, 256, 16, 1, 8, 0.0002)  #epo,lr
    v.train_v_func_inloop(v_func, v, 0.5)  #threshold
    print("used episodes", len(training_data))
    print("average length per trajectory", sum([len(a) for a in training_data])/len(training_data))
    if avr > max(avr_reward):
        print("save policy")
        data_collector.save_weights(weights, weights_var, v, size, index)
        print(size)
    reduce, reset = if_reset(avr, avr_reward, threshold)
    if reset:
        weights, weights_var, size, v, index = data_collector.load_w()
        print("reset both")
    elif reduce:
        l_rate = max(decay*l_rate, lr_min)
        #nn.__init__(tra, l_rate, seed, size, decay=max(0.1, var_decay**epo))
        nn.learning_rate = l_rate, nn.decay_var = max(0.1, var_decay**epo)
        l_rate_var = max(decay*l_rate_var, lr_min)
        #nn_var.__init__(tra, l_rate_var, seed, size_var, decay=max(0.1, var_decay**epo))
        nn_var.learning_rate = l_rate_var, nn_var.decay_var = max(0.1, var_decay ** epo)
        print("decrease learning rate and variance")
        print("variance", max(0.1, var_decay ** epo))
    else:
        if epo % 3 != 0:  #1,2,4,5,7,8..
            weights, size = nn.choose_node(weights, nn, weights_var, nn_var, size, dc, threshold_node, index, v)
            weights, size, index = nn.choose_layer(weights, nn, weights_var, nn_var, size, dc,
                                                   threshold_layer, index, v)
            weights, index = nn.backward_ng2_fast(training_data, actions, R, weights, v, variance, regression, index)
            print("update mean", weights[-1])
        else:  #3,6,9,..
            weights_var = nn_var.backward_ng2_fast_variance(training_data, actions, R, weights_var, v, mean, regression)
            print("update var", weights_var[-1])
    print("best trajectory", best[epo-start])
    print("worst trajectory", worst[epo-start])
    print("size", size)
    print("index", index)
print(max(avr_reward))
print(avr_reward.index(max(avr_reward))+1)
