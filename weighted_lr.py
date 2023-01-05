import torch
import numpy as np
#from execute import v_func,training_data
import copy
import random
class lr(object):
    def __init__(self,importance_part,training_data_nr,bandwidth,sig):#most_import_nr,l_for_kernel,ridge_regre
        self.part = importance_part
        self.l = bandwidth
        self.sigma = sig
        self.nr = training_data_nr

    def choose_v(self,v_func):
        random.shuffle(v_func)
        return v_func[:self.nr]
    
    def weights(self,s,v_func):
        states = torch.cat([a[0] for a in v_func])
        s = s*torch.ones(len(states),5)
        weights = torch.exp(-((sum(((s-states)**2).T))**0.5)/(2*self.l*self.l))
        return weights
    
    def closest_weights(self,weights):
        t = copy.deepcopy(list(weights))
        max_w = []
        max_index = []
        for _ in range(max(3,int(self.part*len(t)))):
            number = max(t)#best reward
            index = t.index(number)#corres. index
            t[index] = 0
            max_w.append(weights[index])
            max_index.append(index)
        return max_w,max_index
    
    def x_tilda(self,v_func,max_index):
        states = torch.cat([a[0] for a in v_func])
        x = [states[i] for i in max_index]
        x_tilda = torch.cat([torch.cat((torch.tensor(1).view([1,1]),a.view([1,5])),1) for a in x])
        return x_tilda
    
    def W(self,max_w):
        max_x = np.array(max_w)
        return torch.tensor(max_x*np.eye(max_x.shape[0]))
    
    def Y(self,v_func,max_index):
        values = torch.cat([a[1].view([1,1]) for a in v_func])
        y = [values[i] for i in max_index]
        return torch.cat(y)
    
    def theta(self,s,v_func):
        v_func = self.choose_v(v_func)
        ws = self.weights(s,v_func)
        max_w,max_index = self.closest_weights(ws)
        x_tilda = self.x_tilda(v_func,max_index)
        w = self.W(max_w)
        y = self.Y(v_func,max_index)
        first_part = torch.mm(torch.mm(x_tilda.T,w.float()),x_tilda)
        second_part = self.sigma*len(v_func)*torch.eye(first_part.shape[0])
        third_part = torch.inverse(first_part+second_part)
        fourth_part = torch.mm(x_tilda.T,w.float())
        theta = torch.mm(torch.mm(third_part,fourth_part),y.view([y.shape[0],1]))
        return theta.T

    def test(self,v_func,regre):
        print("test approximation")
        random.shuffle(v_func)
        test = v_func[0:10]
        train = v_func[50:]
        mse = 0
        for i in range(len(test)):
            the = regre.theta(test[i][0],train)
            s_n = torch.cat([torch.tensor(1).view([1,1]),test[i][0]],1).T
            mse += (test[i][1]-torch.mm(the,s_n))**2
        print("root mean square error",(mse/len(test))**0.5)
        print("predic",torch.mm(the,s_n))
        print("gt",test[i][1])
        return

    def do_adv(self,s_next,s,v_func,regre):
        theta_next = regre.theta(s_next,v_func)
        theta_curr = regre.theta(s,v_func)
        s_n = torch.cat([torch.tensor(1).view([1,1]),s_next],1).T
        s_c = torch.cat([torch.tensor(1).view([1,1]),s],1).T
        adv = float(r) + float(self.decay_adv*torch.mm(theta_next,s_n)) - float(torch.mm(theta_curr,s_c))
        return adv
    
