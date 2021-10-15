import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import biogeme.database as db
from tqdm.auto import tqdm, trange
from time import sleep
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time
import copy
import os
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

def load_data(batch_size=1000):
    X_train_torch = torch.from_numpy(X_train).float()
    y_train_torch  = torch.squeeze(torch.from_numpy(y_train))
    z_train_torch  = torch.squeeze(torch.from_numpy(z_train))

    X_test_torch  = torch.from_numpy(X_test).float()
    y_test_torch  = torch.squeeze(torch.from_numpy(y_test))
    z_test_torch  = torch.squeeze(torch.from_numpy(z_test))

    train_ds = TensorDataset(X_train_torch, y_train_torch, z_train_torch)
    train_dl = DataLoader(train_ds, batch_size, shuffle = False)
    test_ds = TensorDataset(X_test_torch,y_test_torch,z_test_torch)
    test_dl = DataLoader(test_ds, batch_size, shuffle = False)
    
    net = Net(X_train.shape[1])
    
    return train_dl, test_dl, net

def correlation(z,y_pred_class):
    v_yhat=y_pred_class - torch.mean(y_pred_class.float())
    v_z=z-torch.mean(z.float())
    return torch.sum(v_yhat * v_z) / (torch.sqrt(torch.sum(v_yhat ** 2)+1e-20) * torch.sqrt(torch.sum(v_z ** 2)+1e-20))

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   
        os.makedirs(path)  
        
def train_model(train_dl,test_dl,net,lr=0.01, lam=0,num_epochs=10):
    

    criterion=nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(net.parameters(), lr=lr)

    total_train=len(train_dl.dataset)
    total_test=len(test_dl.dataset)

    start_time = time.time() 
    
    train_accuracy_list=[]
    test_accuracy_list=[]
    train_cost1_list=[]
    test_cost1_list=[]
    train_cost2_list=[]
    test_cost2_list=[]
    
    best_model_wts = copy.deepcopy(net.state_dict())
    best_test_accuracy=0.0
    best_cost=100
    best_iter=-1
    last_improvement=0
    require_improvement=8
    stop=False
    epoch=0


    while epoch <num_epochs and stop==False:
        train_predlist=torch.zeros(0,dtype=torch.long)
        train_lbllist=torch.zeros(0,dtype=torch.long)
        train_zlist=torch.zeros(0,dtype=torch.long)
        train_output_list=torch.zeros(0,dtype=torch.long)
        

        train_predlist_0=torch.zeros(0,dtype=torch.long)
        train_lbllist_0=torch.zeros(0,dtype=torch.long)
        train_predlist_1=torch.zeros(0,dtype=torch.long)
        train_lbllist_1=torch.zeros(0,dtype=torch.long)
        
        test_predlist=torch.zeros(0,dtype=torch.long)
        test_lbllist=torch.zeros(0,dtype=torch.long)
        test_zlist=torch.zeros(0,dtype=torch.long)
        test_output_list=torch.zeros(0,dtype=torch.long)
        
        test_output_list=torch.zeros(0,dtype=torch.long)
        test_predlist_0=torch.zeros(0,dtype=torch.long)
        test_lbllist_0=torch.zeros(0,dtype=torch.long)
        test_predlist_1=torch.zeros(0,dtype=torch.long)
        test_lbllist_1=torch.zeros(0,dtype=torch.long)

        running_corrects_train=0
        running_corrects_test=0
        
        for batch_idx, (input_x,label,z) in enumerate(tqdm(train_dl)):
            
            net.train()
      
            y_pred = net(input_x)
          

            cost1 = (1-lam)*criterion(y_pred, label)
            cost2=lam*abs(correlation(torch.masked_select(z,label.bool()),torch.masked_select(y_pred[:,1],label.bool())))
            cost=cost1+cost2
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            train_output_list=torch.cat([train_output_list,y_pred.long()])
            
            m = nn.Softmax(dim=1)
            y_pred=m(y_pred)
            y_pred_class = torch.max(y_pred, 1)[1]

            train_predlist_0_temp=torch.masked_select(y_pred_class.view(-1),~z.bool())
            train_lbllist_0_temp=torch.masked_select(label.view(-1),~z.bool())
            train_predlist_1_temp=torch.masked_select(y_pred_class.view(-1),z.bool())
            train_lbllist_1_temp=torch.masked_select(label.view(-1),z.bool())

            train_predlist_0=torch.cat([train_predlist_0,train_predlist_0_temp])
            train_lbllist_0=torch.cat([train_lbllist_0,train_lbllist_0_temp])
            train_predlist_1=torch.cat([train_predlist_1,train_predlist_1_temp])
            train_lbllist_1=torch.cat([train_lbllist_1,train_lbllist_1_temp])

            train_predlist=torch.cat([train_predlist,y_pred_class.view(-1)])
            train_lbllist=torch.cat([train_lbllist,label.view(-1)])
            train_zlist=torch.cat([train_zlist,z.long().view(-1)])

            running_corrects_train += torch.sum(y_pred_class == label)
            
        train_cm1=confusion_matrix(train_lbllist_1.numpy(), \
                               train_predlist_1.numpy()).ravel()
        train_cm0=confusion_matrix(train_lbllist_0.numpy(), \
                                   train_predlist_0.numpy()).ravel()
        
        train_FNR0=train_cm0[2]/(train_cm0[2]+train_cm0[3])
        train_FNR1=train_cm1[2]/(train_cm1[2]+train_cm1[3])
        train_FNR_gap=train_FNR0-train_FNR1

        train_cost1_epoch=(1-lam)*criterion(train_output_list.float(), train_lbllist)
        train_cost2_epoch=lam*abs(correlation(torch.masked_select(train_zlist,train_lbllist.bool()),torch.masked_select(train_predlist,train_lbllist.bool())))
        train_cost_epoch=train_cost1_epoch+train_cost2_epoch
        train_cost1_epoch=train_cost1_epoch.numpy()
        train_cost2_epoch=train_cost2_epoch.numpy()
        train_cost_epoch=train_cost_epoch.numpy()
        
        
        train_cost1_list.append(train_cost1_epoch)
        train_cost2_list.append(train_cost2_epoch)
        
        train_accuracy=running_corrects_train.double()/total_train
        train_accuracy=train_accuracy.numpy()
        train_accuracy_list.append(train_accuracy)

        for batch_idx, (input_x, label,z) in enumerate(tqdm(test_dl)):

            y_pred = net.eval()(input_x)
            test_output_list=torch.cat([test_output_list,y_pred.long()])
            m = nn.Softmax(dim=1)
            y_pred=m(y_pred)
            y_pred_class = torch.max(y_pred, 1)[1]

            running_corrects_test += torch.sum(y_pred_class == label)

            test_predlist_0=torch.cat([test_predlist_0,torch.masked_select(y_pred_class.view(-1),~z.bool())])
            test_lbllist_0=torch.cat([test_lbllist_0,torch.masked_select(label.view(-1),~z.bool())])
            test_predlist_1=torch.cat([test_predlist_1,torch.masked_select(y_pred_class.view(-1),z.bool())])
            test_lbllist_1=torch.cat([test_lbllist_1,torch.masked_select(label.view(-1),z.bool())])
            
            test_predlist=torch.cat([test_predlist,y_pred_class.view(-1)])
            test_lbllist=torch.cat([test_lbllist,label.view(-1)])
            test_zlist=torch.cat([test_zlist,z.long().view(-1)])

        test_cm1=confusion_matrix(test_lbllist_1.numpy(), \
                               test_predlist_1.numpy()).ravel()
        test_cm0=confusion_matrix(test_lbllist_0.numpy(), \
                                   test_predlist_0.numpy()).ravel()

        test_FNR0=test_cm0[2]/(test_cm0[2]+test_cm0[3])
        test_FNR1=test_cm1[2]/(test_cm1[2]+test_cm1[3])
        test_FNR_gap=test_FNR0-test_FNR1
        
        test_cost1_epoch=(1-lam)*criterion(test_output_list.float(), test_lbllist)
        test_cost2_epoch=lam*abs(correlation(torch.masked_select(test_zlist,test_lbllist.bool()),torch.masked_select(test_predlist,test_lbllist.bool())))
        test_cost_epoch=test_cost1_epoch+test_cost2_epoch
        
        test_cost1_epoch=test_cost1_epoch.numpy()
        test_cost2_epoch=test_cost2_epoch.numpy()
        test_cost_epoch=test_cost_epoch.numpy()
        
        test_cost1_list.append(test_cost1_epoch)
        test_cost2_list.append(test_cost2_epoch)
        
        test_accuracy=running_corrects_test.double()/total_test
        test_accuracy=test_accuracy.numpy()
        test_accuracy_list.append(test_accuracy)

        print("Epoch {}: Training Accuracy {}; Cost1 {}; Cost2 {}; Cost {}".format(epoch, np.round(train_accuracy,6), \
                                                                                   np.round(float(train_cost1_epoch),6),\
                                                                                   np.round(float(train_cost2_epoch),6),\
                                                                                   np.round(float(train_cost_epoch),6))) 
        print("Train_FNR_gap {}; Test_FNR_gap {}".format(np.round(train_FNR_gap,6),np.round(test_FNR_gap,6)))

        if train_cost_epoch< best_cost:
            best_cost = train_cost_epoch
            best_model_wts = copy.deepcopy(net.state_dict())
            best_iter=epoch
            last_improvement=0
        else:
            last_improvement+=1
            
        if last_improvement > require_improvement:
            print("No improvement found during the "+str(require_improvement)+" last iterations, stopping optimization.")
            # Break out from the loop.
            stop = True
            
        epoch+=1
        
    print("best_iter:",best_iter)
    final_iter=epoch

    
    return best_model_wts,train_accuracy_list,test_accuracy_list,\
            train_cost1_list,test_cost1_list,train_cost2_list,test_cost2_list,best_iter,final_iter

def evaluate_model(best_model_wts,t_dl,lam):
    print('Evaluating...')
    net = Net(X_train.shape[1])
    net.load_state_dict(best_model_wts)
    
    criterion=nn.CrossEntropyLoss(reduction='mean')

    total_test=len(t_dl.dataset)
    test_predlist=torch.zeros(0,dtype=torch.long)
    test_lbllist=torch.zeros(0,dtype=torch.long)
    test_zlist=torch.zeros(0,dtype=torch.long)
    test_output_list=torch.zeros(0,dtype=torch.long)

    test_output_list=torch.zeros(0,dtype=torch.long)
    test_predlist_0=torch.zeros(0,dtype=torch.long)
    test_lbllist_0=torch.zeros(0,dtype=torch.long)
    test_predlist_1=torch.zeros(0,dtype=torch.long)
    test_lbllist_1=torch.zeros(0,dtype=torch.long)
    running_corrects_test=0
    
    with torch.no_grad():
        for batch_idx, (input_x, label,z) in enumerate(tqdm(t_dl)):

            y_pred = net.eval()(input_x)

            test_output_list=torch.cat([test_output_list,y_pred.long()])
            m = nn.Softmax(dim=1)
            y_pred=m(y_pred)
            y_pred_class = torch.max(y_pred, 1)[1]


            running_corrects_test += torch.sum(y_pred_class == label)

            test_predlist_0=torch.cat([test_predlist_0,torch.masked_select(y_pred_class.view(-1),~z.bool())])
            test_lbllist_0=torch.cat([test_lbllist_0,torch.masked_select(label.view(-1),~z.bool())])
            test_predlist_1=torch.cat([test_predlist_1,torch.masked_select(y_pred_class.view(-1),z.bool())])
            test_lbllist_1=torch.cat([test_lbllist_1,torch.masked_select(label.view(-1),z.bool())])
            
            test_predlist=torch.cat([test_predlist,y_pred_class.view(-1)])
            test_lbllist=torch.cat([test_lbllist,label.view(-1)])
            test_zlist=torch.cat([test_zlist,z.long().view(-1)])

        test_cm0=confusion_matrix(test_lbllist_0.numpy(), \
                           test_predlist_0.numpy()).ravel()
        test_cm1=confusion_matrix(test_lbllist_1.numpy(), \
                               test_predlist_1.numpy()).ravel()
        
        test_cost1_epoch=(1-lam)*criterion(test_output_list.float(), test_lbllist)
        test_cost2_epoch=lam*abs(correlation(torch.masked_select(test_zlist,test_lbllist.bool()),torch.masked_select(test_predlist,test_lbllist.bool())))
        test_cost_epoch=test_cost1_epoch+test_cost2_epoch
        
        test_accuracy=running_corrects_test.double()/total_test
        test_accuracy=test_accuracy.numpy()
        
    return test_accuracy,test_cm0,test_cm1