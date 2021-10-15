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
from train_pytorch import *

Z_var='z'
folder_data='func_quadratic'
folder_model='func_quadratic_torch'
folder='func_quadratic_torch'
model_path ='model_3lyr_'

#define the hyperparameters for the synthetic experiments
n_samples=1000000
n_x=5
cov=0.5
n_seed=5
conv_list=[0,0.25,0.5,0.75,1]
lam_list=list(np.array(range(1,10))/10)

#defining the training hyperparameters
num_models = 5
n_splits=5
batch_size=1000
num_epochs=30
lr=0.001

start_time = time.time()
for cov in conv_list:

    df = []
    data_key='cov'+str(cov)+'_size'+str(n_samples)+'_var'+str(n_x)
    
    for lam in lam_list:
        p_dic={}
        p_dic['train_accuracy']=[]
        p_dic['test_accuracy']=[]
        p_dic['train_cm0']=[]
        p_dic['train_cm1']=[]
        p_dic['test_cm0']=[]
        p_dic['test_cm1']=[]
        p_dic['train_FNR_gap']=[]
        p_dic['test_FNR_gap']=[]

        p_dic['TRAINING_PROCESS']={}
        p_dic['TRAINING_PROCESS']['best_iter']=[]
        p_dic['TRAINING_PROCESS']['final_iter']=[]
        p_dic['TRAINING_PROCESS']['accuracy']={}
        p_dic['TRAINING_PROCESS']['cost1']={}
        p_dic['TRAINING_PROCESS']['cost2']={}
        p_dic['TRAINING_PROCESS']['accuracy']['training_set']=[]
        p_dic['TRAINING_PROCESS']['accuracy']['testing_set']=[]
        p_dic['TRAINING_PROCESS']['cost1']['training_set']=[]
        p_dic['TRAINING_PROCESS']['cost1']['testing_set']=[]
        p_dic['TRAINING_PROCESS']['cost2']['training_set']=[]
        p_dic['TRAINING_PROCESS']['cost2']['testing_set']=[]
            
        for seed in range(n_seed):
            with open('data/sync/'+ folder_data+ '/data_'+data_key+'_seed'+str(seed)+'.pkl', 'rb') as f:
                input_data = pickle.load(f)

            DIR = 'models/sync/' + folder_model+ '/'+model_path+data_key+'_seed'+str(seed)+'_lambda'+str(lam)+ '/'
            
            mkdir(DIR)  

            X_all=input_data['X_all']
            Y_all=input_data['Y_all']

            #final results

            accuracy_train_list_seed = []
            accuracy_test_list_seed = []

            train_cm0_trial_seed=[]
            train_cm1_trial_seed=[]
            test_cm0_trial_seed=[]
            test_cm1_trial_seed=[]

            train_FNR_gap_list_seed=[]
            test_FNR_gap_list_seed=[]


            #training process
            best_iter_seed=[]
            final_iter_seed=[]
            accuracy_training_prcoess_seed=[]
            accuracy_testing_prcoess_seed=[]
            cost1_training_prcoess_seed=[]
            cost1_testing_prcoess_seed=[]
            cost2_training_prcoess_seed=[]
            cost2_testing_prcoess_seed=[]


            for i in range(num_models):
                kf = KFold(n_splits, shuffle=True,random_state=0)

                #final results
                accuracy_train_list = []
                accuracy_test_list = []

                train_cm0_trial=[]
                train_cm1_trial=[]
                test_cm0_trial=[]
                test_cm1_trial=[]

                train_FNR_gap_list=[]
                test_FNR_gap_list=[]

                #training process
                best_iter_trial=[]
                final_iter_trial=[]
                accuracy_training_prcoess_trial=[]
                accuracy_testing_prcoess_trial=[]
                cost1_training_prcoess_trial=[]
                cost1_testing_prcoess_trial=[]
                cost2_training_prcoess_trial=[]
                cost2_testing_prcoess_trial=[]

                con=0
                for train_idx, test_idx in kf.split(X_all,Y_all):
                    con+=1
                    MODEL_NAME = 'model' + str(i)+'_fold'+str(con)
                    X_train = np.array(X_all.iloc[train_idx,:])
                    y_train = np.array(Y_all[train_idx])
                    X_test = np.array(X_all.iloc[test_idx,:])
                    y_test = np.array(Y_all[test_idx])
                    z_train = np.array(X_all.iloc[train_idx,]['z'])
                    z_test = np.array(X_all.iloc[test_idx,]['z'])
                    
                    train_dl, test_dl, net=load_data(batch_size)
                    
                    best_model_wts,train_accuracy_list,test_accuracy_list,\
                    train_cost1_list,test_cost1_list,train_cost2_list,\
                    test_cost2_list,best_iter,final_iter=train_model(train_dl,test_dl,net, lr,lam,num_epochs)
                    print("------------------------------")
                    train_accuracy,train_cm0,train_cm1=evaluate_model(best_model_wts,train_dl,lam)
                    test_accuracy,test_cm0,test_cm1=evaluate_model(best_model_wts,test_dl,lam)
                    print(train_cm0)

                    
                    print('cov:',cov,'; seed:',seed,' ',MODEL_NAME,' completed')
                    print('accuracy_train: ',train_accuracy,' ;accuracy_test: ',test_accuracy)


                    
                    net = Net(X_train.shape[1])
                    net.load_state_dict(best_model_wts)
                    torch.save(net.state_dict(), DIR+MODEL_NAME+'.pth')


                    #save results
                    accuracy_train_list.append(train_accuracy)
                    accuracy_test_list.append(test_accuracy)

                    train_cm0_trial.append(train_cm0)
                    train_cm1_trial.append(train_cm1)

                    test_cm0_trial.append(test_cm0)
                    test_cm1_trial.append(test_cm1)

                    train_FNR0=train_cm0[2]/(train_cm0[2]+train_cm0[3])
                    train_FNR1=train_cm1[2]/(train_cm1[2]+train_cm1[3])
                    train_FNR_gap=train_FNR0-train_FNR1

                    test_FNR0=test_cm0[2]/(test_cm0[2]+test_cm0[3])
                    test_FNR1=test_cm1[2]/(test_cm1[2]+test_cm1[3])
                    test_FNR_gap=test_FNR0-test_FNR1

                    train_FNR_gap_list.append(train_FNR_gap)
                    test_FNR_gap_list.append(test_FNR_gap)
                    
                    print('train_FNR_gap: ',train_FNR_gap,' ;test_FNR_gap: ',test_FNR_gap)

                    time_dur=round(time.time() - start_time, 3)
                    print("--- %s seconds ---" % (time.time() - start_time))
                    print("  ")

                    #save process
                    best_iter_trial.append(best_iter)
                    final_iter_trial.append(final_iter)
                    accuracy_training_prcoess_trial.append(train_accuracy_list)
                    accuracy_testing_prcoess_trial.append(test_accuracy_list)
                    cost1_training_prcoess_trial.append(train_cost1_list)
                    cost1_testing_prcoess_trial.append(test_cost1_list)
                    cost2_training_prcoess_trial.append(train_cost2_list)
                    cost2_testing_prcoess_trial.append(test_cost2_list)
                    
                    if model_path =='model_3lyr_':
                        layer=3
                    else:
                        layer=0

                    df.append([lam,seed, i,con,cov,n_samples,n_x,layer, num_epochs, batch_size,\
                    train_accuracy, test_accuracy,\
                    train_FNR_gap,test_FNR_gap])

                accuracy_train_list_seed.append(accuracy_train_list)
                accuracy_test_list_seed.append(accuracy_test_list)
                train_cm0_trial_seed.append(train_cm0_trial)
                train_cm1_trial_seed.append(train_cm1_trial)
                test_cm0_trial_seed.append(test_cm0_trial)
                test_cm1_trial_seed.append(test_cm1_trial)
                train_FNR_gap_list_seed.append(train_FNR_gap_list)
                test_FNR_gap_list_seed.append(test_FNR_gap_list)

                best_iter_seed.append(best_iter_trial)
                final_iter_seed.append(final_iter_trial)
                accuracy_training_prcoess_seed.append(accuracy_training_prcoess_trial)
                accuracy_testing_prcoess_seed.append(accuracy_testing_prcoess_trial)
                cost1_training_prcoess_seed.append(cost1_training_prcoess_trial)
                cost1_testing_prcoess_seed.append(cost1_testing_prcoess_trial)
                cost2_training_prcoess_seed.append(cost2_training_prcoess_trial)
                cost2_testing_prcoess_seed.append(cost2_testing_prcoess_trial)

            p_dic['train_accuracy'].append(accuracy_train_list_seed)
            p_dic['test_accuracy'].append(accuracy_test_list_seed)
            p_dic['train_cm0'].append(train_cm0_trial_seed)
            p_dic['train_cm1'].append(train_cm1_trial_seed)
            p_dic['test_cm0'].append(test_cm0_trial_seed)
            p_dic['test_cm1'].append(test_cm1_trial_seed)
            p_dic['train_FNR_gap'].append(train_FNR_gap_list_seed)
            p_dic['test_FNR_gap'].append(test_FNR_gap_list_seed)

            p_dic['TRAINING_PROCESS']['accuracy']['training_set'].append(accuracy_training_prcoess_seed)
            p_dic['TRAINING_PROCESS']['accuracy']['testing_set'].append(accuracy_testing_prcoess_seed)
            p_dic['TRAINING_PROCESS']['cost1']['training_set'].append(cost1_training_prcoess_seed)
            p_dic['TRAINING_PROCESS']['cost1']['testing_set'].append(cost1_testing_prcoess_seed)
            p_dic['TRAINING_PROCESS']['cost2']['training_set'].append(cost2_training_prcoess_seed)
            p_dic['TRAINING_PROCESS']['cost2']['testing_set'].append(cost2_testing_prcoess_seed)

        with open('results/sync/' + folder+ '/'+'performance_'+model_path+data_key+'_lambda'+str(lam)+'.pkl', 'wb') as f:
            pickle.dump(p_dic, f)

        df = pd.DataFrame(df, columns = ['lambda','seed','trial','fold','cov','n_samples','n_x','#layers','#iteration','batch_size','accuracy_train','accuracy_test','FNR_gap_train','FNR_gap_test'])
        df.to_csv('results/sync/' + folder+ '/'+'performance_'+model_path+data_key+'_lambda'+str(lam)+ '.csv')