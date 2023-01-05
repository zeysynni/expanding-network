import numpy as np
import torch
import torch.nn.functional as F
from weighted_lr import lr
import copy
from copy import deepcopy
#defines the policy model which is a nn
#seed = 0
#torch.manual_seed(seed)
class AdamOptim():
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        #self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
    def update(self, t, w, dw):
        ## dw, db are from current minibatch
        ## momentum beta 1
        # *** weights *** #
        if t == 1:
            self.m_dw = [self.m_dw * torch.ones(dw[i].shape) for i in range(len(dw))]
        #print("mdw",self.m_dw)
        #assert not torch.isnan(self.m_dw[3]).any(), print(self.m_dw[3])
        self.m_dw = [self.beta1*self.m_dw[i] + (1-self.beta1)*dw[i] for i in range(len(dw))]
        #assert not torch.isnan(self.m_dw[3]).any(), print("dw", dw[3])
        #print("m_dw", self.m_dw[3])
        # *** biases *** #
        #self.m_db = self.beta1*self.m_db + (1-self.beta1)*db

        ## rms beta 2
        # *** weights *** #
        if t == 1:
            self.v_dw = [self.v_dw * torch.ones(dw[i].shape) for i in range(len(dw))]
        self.v_dw = [self.beta2*self.v_dw[i] + (1-self.beta2)*(dw[i]**2) for i in range(len(dw))]
        #assert not torch.isnan(self.v_dw[3]).any(), print(self.v_dw[3])
        #print("v_dw", self.v_dw[3])
        # *** biases *** #
        #self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db)

        ## bias correction
        m_dw_corr = [self.m_dw[i]/(1-self.beta1**t) for i in range(len(dw))]
        #assert not torch.isnan(m_dw_corr[3]).any(), print(self.m_dw_corr[3])
        #m_db_corr = self.m_db/(1-self.beta1**t)
        v_dw_corr = [self.v_dw[i]/(1-self.beta2**t) for i in range(len(dw))]
        #assert not torch.isnan(v_dw_corr[3]).any(), print(self.v_dw_corr[3])
        #print("m_dv corr", m_dw_corr[3])
        #print("v_dv corr", v_dw_corr[3])
        #v_db_corr = self.v_db/(1-self.beta2**t)

        ## update weights and biases
        w = [(w[i] + self.eta*(m_dw_corr[i]/(np.sqrt(v_dw_corr[i])+self.epsilon))).requires_grad_(True)
             for i in range(len(dw))]
        """
        assert not torch.isnan(w[3]).any(), print("np.sqrt(v_dw_corr[i])", np.sqrt(v_dw_corr[3]),
                                                  "(np.sqrt(v_dw_corr[i])+self.epsilon)",
                                                  (np.sqrt(v_dw_corr[3]) + self.epsilon),
                                                  "m_dw_corr[i]", m_dw_corr[3])
        """
        #b = b - self.eta*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
        return w#, b
    def update_for_activatoin(self, t, w, dw):
        ## dw, db are from current minibatch
        ## momentum beta 1
        # *** weights *** #
        #only a 3-dim tensor 
        if t == 1:
            self.m_dw = self.m_dw * torch.ones(dw.shape)
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        ## rms beta 2
        # *** weights *** #
        if t == 1:
            self.v_dw = self.v_dw * torch.ones(dw.shape)
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
        ## bias correction
        m_dw_corr = self.m_dw/(1-self.beta1**t)
        v_dw_corr = self.v_dw/(1-self.beta2**t)
        ## update weights and biases
        w = (w + self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))).requires_grad_(True)
        return w
    def if_converge(self,w_old,w_new):
        return sum([torch.norm((w_old[i]-w_new[i]), p="fro") for i in range(len(w_new))])

class mlp(object):
    #torch.manual_seed(self.seed)
    def __init__(self,trajectories,rate,seed,size,decay):
        self.learning_rate = rate
        self.decay_adv = 0.97
        self.network_size = size#size should be a list
        self.decay_var = decay
        self.seed = seed
        self.traj = trajectories
    def weights(self):
        torch.manual_seed(self.seed)
        weights = []
        for weight in self.network_size:
            weights.append((torch.randn(weight)/np.sqrt(weight[0])).requires_grad_(True))
        return weights
    def weights_var(self):#he initialization
        torch.manual_seed(self.seed)
        weights = []
        for weight in self.network_size:
            n = weight[0]
            std = np.sqrt(2.0/n)
            weights.append((torch.randn(weight)*std).requires_grad_(True))
        return weights
    def forward(self,x,weights,index):
        for w in enumerate(weights):
            if w[0] == 0:
                out = torch.mm(x,w[1])
                if str(w[0]) in index.keys():
                    #w[0] is a list of one single tensor,w[0][0] is the tensor, w[0][0][0] is the first term of tensor
                    #out = index[str(w[0])][0][0]*out +(index[str(w[0])][0][1] + index[str(w[0])][0][2]*out)/(1 + out**2)
                    out = index[str(w[0])][0]*out + (index[str(w[0])][1] + index[str(w[0])][2]*out)/(1 + out**2)    
                else:
                    out = torch.tanh(out)
            else:
                out = torch.mm(out,w[1])
                if str(w[0]) in index.keys():
                    out = index[str(w[0])][0]*out + (index[str(w[0])][1] + index[str(w[0])][2]*out)/(1 + out**2)
                    #out = index[str(w[0])][0][0]*out +(index[str(w[0])][0][1] + index[str(w[0])][0][2]*out)/(1 + out**2)
                else:
                    out = torch.tanh(out)
        return out
    def forward_var(self,x,weights):
        for w in enumerate(weights):
            if w[0] == 0:
                out = torch.mm(x,w[1])
                #out = torch.tanh(out)
                out = F.relu(out)
            elif w[0] == len(weights)-1:
                out = torch.mm(out,w[1])
                out = torch.sigmoid(out)
                #out = F.relu(out)
            else:
                out = torch.mm(out,w[1])
                #out = torch.tanh(out)
                out = F.relu(out)
        return self.decay_var*out
    def size_info(self,weights,nn,weights_var,nn_var,dc,index,v):
        #compute ita value for one position
        training_data, R, actions, mean, variance = dc.letsgobrandon1(weights,nn,weights_var,nn_var,index)
        v_func = dc.train_data_v(training_data,R)
        v.train_v_func_inloop(v_func,v,0.5)
        fisher,gs = nn.F_and_g(training_data,actions,weights,variance,index)
        gs = (torch.mean(gs,dim=1)).view([1,-1])
        ita = torch.mm(torch.mm(gs,fisher),gs.T)
        return ita
    def choose_node(self,weights,nn,weights_var,nn_var,size,dc,thres,index,v,x=0):
        #if need to expand a layer size
        w = deepcopy(weights)
        ratio = []
        w_init_all = []#input weights
        w_init_all_out = []#output weights
        ita_init = self.size_info(weights,nn,weights_var,nn_var,dc,index,v)#/sum([i[0]*i[1] for i in size])
        for init in range(10):
            ita = []
            w_init = []
            w_init_out = []
            for pos in enumerate(size[:-1]):
                add_1 = (torch.randn([pos[1][0],1])/np.sqrt(pos[1][0]))
                add_2 = (torch.zeros([1,size[pos[0]+1][1]]))
                weights[pos[0]] = (torch.cat((weights[pos[0]].requires_grad_(False),add_1),1)).requires_grad_(True)
                weights[pos[0]+1] = (torch.cat((weights[pos[0]+1].requires_grad_(False),add_2),0)).requires_grad_(True)
                ita.append(self.size_info(weights,nn,weights_var,nn_var,dc,index,v))
                w_init.append(deepcopy(weights[pos[0]]))
                w_init_out.append(deepcopy(weights[pos[0]+1]))
                weights = deepcopy(w)
            ratio.append([a/ita_init for a in ita])
            w_init_all.append(w_init)
            w_init_all_out.append(w_init_out)
        for i in enumerate(ratio):
            if x < max(i[1]):
                x = max(i[1])
                loc = i[0]#which init
                index = i[1].index(x)#which pos
        print("radio to init",x)
        if x > thres:#10
            w[index]=w_init_all[loc][index]
            w[index+1]=w_init_all_out[loc][index]
            size[index][1] += 1
            size[index+1][0] += 1
            print("change size",size)
        else:
            print("no change in size",size)
        return w,size
    def choose_layer(self,weights,nn,weights_var,nn_var,size,dc,thres,index,v,x=0):
        #if need to add a hidden layer
        #w = [deepcopy(weight).clone().detach() for weight in weights]
        w = deepcopy(weights)
        index_ori = deepcopy(index)
        ratio = []
        w_init_all = []#input weights
        w_init_all_out = []#output weights
        ita_init = self.size_info(weights,nn,weights_var,nn_var,dc,index,v)#/sum([i[0]*i[1] for i in size])
        for init in range(1):
            ita = []
            w_init = []
            w_init_out = []
            for pos in enumerate(size):
                #die 1. Matrix durch zwei ersetzen
                lp = (torch.randn([pos[1][0],pos[1][0]])/np.sqrt(pos[1][0]**2)).requires_grad_(True)
                lp_2 = (torch.mm(torch.inverse(lp).clone().detach(),weights[pos[0]].clone().detach())).requires_grad_(True)
                weights[pos[0]] = lp
                weights.insert(pos[0]+1,lp_2)
                #added layer position and initial activation function parameters
                #the index number is the hidden layer number
                index[str(pos[0])] = torch.tensor((1.,0.,0.)).requires_grad_(True)
                ita.append(self.size_info(weights,nn,weights_var,nn_var,dc,index,v))
                w_init.append(deepcopy(lp))
                w_init_out.append(deepcopy(lp_2))
                weights = deepcopy(w)#[weight.requires_grad_(True) for weight in w]
                index = deepcopy(index_ori)
            ratio.append([a/ita_init for a in ita])
            #append for each init
            w_init_all.append(w_init)
            w_init_all_out.append(w_init_out)
        #find out the biggest ratio & corres. init & location
        for i in enumerate(ratio):
            #for each init, check the biggest ita value for all the locations
            if x < max(i[1]):
                x = max(i[1])
                weight_init = i[0]#which init
                layer = i[1].index(x)#which pos
        print("layer ratio",x)
        if x > thres:
            #change size
            size.insert(layer,[size[layer][0],size[layer][0]])
            #add new weights
            w[layer]=w_init_all[weight_init][layer]
            w.insert(layer+1,w_init_all_out[weight_init][layer])
            #add activation func param
            #if the to be andded hidden layer is at the beginning, later hidden layer index should be added by 1
            keys = [i for i in index.keys()]
            for i in range(len(keys)):
                if layer <= int(keys[i]):
                    index_ori[str(int(keys[i])+1)] = index_ori.pop(keys[i])
            index_ori[str(layer)] = torch.tensor((1.,0.,0.)).requires_grad_(True)
            print("added hidden layer",layer)
        else:
            print("no change in hidden layer")
        return w,size,index_ori
    def fisher_matrix(self,grads,gs):
        #grad with adv/grad wo. adv
        for i in range(len(gs)):
            gs[i] = torch.cat([a for a in gs[i]],0)
        gs = (torch.cat([a for a in gs],1)).clone().detach()
        form = []
        #save cumulated grad for each weight matrix
        for i in range(len(grads)):
            form.append(grads[i].shape)
            #vectorize weights matrix
            grads[i] = grads[i].reshape(-1,1)
        #combine all the vectors together
        dtheta_re = torch.cat(grads,0)
        #print("len",len(grads))
        #print("dtheta shape",dtheta_re.shape)
        fisher = torch.mm(gs,gs.T)/self.traj
        #print("fisher form",fisher.shape)
        #print("rank gs",torch.linalg.matrix_rank(gs))
        #print("form gs",gs.shape)
        #print(gs)
        #print("rank fisher",torch.linalg.matrix_rank(fisher,hermitian=True))
        return fisher,form,dtheta_re,gs
    def fisher_for_activation(self,grads,gs):
        #grad with adv/grad wo. adv
        for i in range(len(gs)):
            gs[i] = torch.cat([a for a in gs[i]],0)
        gs = (torch.cat([a for a in gs],1)).clone().detach()
        dtheta_re = grads
        fisher = torch.mm(gs,gs.T)/self.traj
        return fisher,dtheta_re
    def cal_update(self,fisher,form,dtheta_re):
        update = torch.linalg.lstsq(fisher,dtheta_re).solution
        #print("pinv")
        updates = []
        nr = 0
        for f in enumerate(form):
            if f[0] == 0:
                nr_new = f[1][0]*f[1][1]
                updates.append(update[:nr+nr_new].view(f[1]))
                nr += nr_new
            else:
                nr_new = f[1][0]*f[1][1]
                updates.append(update[nr:nr+nr_new].view(f[1]))
                nr += nr_new
        return updates
    def backward_ng2_fast(self,training_data,actions,R,weights,value,variance,regre,index):
        #use ng
        #use sum of each episode, do average over averages
        #with each set of episodes, train the policy on it
        #optimizer = torch.optim.Adam(weights, lr=self.lr)
        print("lr",self.learning_rate)
        #for epoch in range(1,self.n_iters):
        converged = False
        epoch = 1
        adam = AdamOptim(eta=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        adam_activation = AdamOptim(eta=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        improve = 1
        while not converged:
            #print("epoch",epoch)
            #average of grad over all episode
            #amout of grad for each episode
            #go through every episode,len is how many episodes we use
            w_old = copy.deepcopy(weights)
            w_adv = []
            [w_adv.append(0) for i in weights]
            #grad with adv
            layer_para = {}
            for key in index.keys():
                layer_para[key] = 0
            grads = []
            #grad wo adv
            layer_grad = {}
            for key in index.keys():
                layer_grad[key] = []
            for epi in range(len(training_data)):
                s = torch.cat([a for a in training_data[epi]])
                s_next = torch.cat([s[torch.arange(s.size(0))!=0],torch.tensor([[0,0,0,0,0]]).float()])
                s_value = value(s)
                s_next_value = value(s_next)
                r = torch.cat([torch.tensor(a).view([1,1]) for a in R[epi]])
                a = torch.cat([a.view([1,1]) for a in actions[epi]])
                adv = r + self.decay_adv*s_next_value - s_value
                var = torch.cat([torch.tensor(a).view([1,1]) for a in variance[epi]])
                a4 = self.forward(s,weights,index)
                adv_grad_1 = (torch.tensor([float(b) for b in (a-a4)/(var**2)])).view([-1,1])
                adv_grad_2 = adv_grad_1*a4*adv
                sum(adv_grad_2).backward()
                w_adv = [w_adv[i]+weights[i].grad for i in range(len(weights))]
                [weights[a].grad.zero_() for a in range(len(weights))]
                for key in index.keys():
                    layer_para[key] += index[key].grad
                    index[key].grad.zero_()
                a4 = self.forward(s,weights,index)
                sum(adv_grad_1*a4).backward()
                grads.append([(copy.deepcopy(weights[a].grad)).view(-1,1) for a in range(len(weights))])
                [weights[a].grad.zero_() for a in range(len(weights))]
                for key in index.keys():
                    layer_grad[key].append([(copy.deepcopy(index[key].grad)).view(-1,1)])
                    index[key].grad.zero_()

            #average over trajectories
            grad_adv = [a/len(training_data) for a in w_adv]
            fisher,form,dtheta_re,gs = self.fisher_matrix(grad_adv,grads)
            with torch.no_grad():
                #update list of matrix for weights
                updates = self.cal_update(fisher,form,dtheta_re)
                #use adam to optimize
                weights = adam.update(epoch, weights, updates)
                #update for activation
                for key in index.keys():
                    layer_para[key] = layer_para[key]/len(training_data)
                    #fisher matrix/grad_adv vector
                    f,d_r = self.fisher_for_activation(layer_para[key],layer_grad[key])
                    update_activation = torch.linalg.lstsq(f,d_r).solution
                    #update the parameter for each hidden layer seperately
                    index[key] = adam_activation.update_for_activatoin(epoch, index[key], update_activation)
            current_change = adam.if_converge(w_old,weights)
            if current_change < 0.1 and current_change > improve:
                print("converged")
                break
            elif epoch > 200 or torch.norm(updates[-1], p="fro") < 0.000001:
                print("maximal iteration exceed")
                break
            else:
                improve = adam.if_converge(w_old, weights)
                epoch += 1
        return weights,index

    def backward_ng2_fast_variance(self,training_data,actions,R,weights_var,value,mean,regre):
        #use ng
        #use average over each episode, do average over averages
        #with each set of episodes, train the policy to it
        print("lr",self.learning_rate)
        converged = False
        epoch = 1
        adam = AdamOptim(eta=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        improve = 1
        #for epoch in range(self.n_iters):
        while not converged:
            #average of grad over all episode
            #amout of grad for each episode
            #go through every episode,len is how many episodes we use
            w_old = copy.deepcopy(weights_var)
            w_var_adv = []
            [w_var_adv.append(0) for i in weights_var]
            grads = []
            for epi in range(len(training_data)):
                s = torch.cat([a for a in training_data[epi]])
                s_next = torch.cat([s[torch.arange(s.size(0))!=0],torch.tensor([[0,0,0,0,0]]).float()])
                s_value = value(s)
                s_next_value = value(s_next)
                r = torch.cat([torch.tensor(a).view([1,1]) for a in R[epi]])
                a = torch.cat([a.view([1,1]) for a in actions[epi]])
                a4 = torch.cat([a.view([1,1]) for a in mean[epi]])
                adv = r + self.decay_adv*s_next_value - s_value
                var = self.forward_var(s,weights_var)
                cons_step = (torch.tensor([float(a) for a in 2*np.pi*var+((a-a4)**2)/(var**3)])).view([-1,1])
                adv_grad_2 = cons_step*var*adv
                sum(adv_grad_2).backward()
                w_var_adv = [w_var_adv[i]+weights_var[i].grad for i in range(len(weights_var))]
                #w_var_adv = [w_var_adv[i]+copy.deepcopy(weights_var[i].grad) for i in range(len(weights_var))]
                [weights_var[a].grad.zero_() for a in range(len(weights_var))]
                var = self.forward_var(s,weights_var)
                sum(cons_step*var).backward()
                grads.append([(copy.deepcopy(weights_var[a].grad)).view(-1,1) for a in range(len(weights_var))])
                #grads.append([(copy.deepcopy(weights_var[a].grad)).view(1,-1) for a in range(len(weights_var))])
                [weights_var[a].grad.zero_() for a in range(len(weights_var))]
            grad_adv = [a/len(training_data) for a in w_var_adv]
            fisher,form,dtheta_re,gs = self.fisher_matrix(grad_adv,grads)
            with torch.no_grad():
                updates = self.cal_update(fisher,form,dtheta_re)
                weights_var = adam.update(epoch, weights_var, updates)
            current_change = adam.if_converge(w_old,weights_var)
            if current_change < 0.1 and current_change > improve:
                print("converged")
                break
            elif epoch > 200 or torch.norm(updates[-1], p="fro") < 0.000001:
                print("maximal iteration exceed")
                break
            else:
                improve = adam.if_converge(w_old, weights_var)
                epoch += 1
        return weights_var
    
    def F_and_g(self,training_data,actions,weights,variance,index):
        #use ng
        #use sum of each episode, do average over averages
        #with each set of episodes, train the policy on it
        grads = []
        for epi in range(len(training_data)):
            s = torch.cat([a for a in training_data[epi]])
            a = torch.cat([a.view([1,1]) for a in actions[epi]])
            var = torch.cat([torch.tensor(a).view([1,1]) for a in variance[epi]])
            a4 = self.forward(s,weights,index)
            adv_grad_1 = (torch.tensor([float(b) for b in (a-a4)/(var**2)])).view([-1,1])
            sum(adv_grad_1*a4).backward()
            grads.append([(copy.deepcopy(weights[a].grad)).view(-1,1) for a in range(len(weights))])
            [weights[a].grad.zero_() for a in range(len(weights))]
        for i in range(len(grads)):
            grads[i] = torch.cat([a for a in grads[i]],0)
        gs = (torch.cat([a for a in grads],1)).clone().detach()
        fisher = torch.mm(gs,gs.T)/self.traj
        return fisher, gs#weights,fisher,gs
