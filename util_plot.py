import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import os
import copy
import tensorflow as tf
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import time
import collections
from sklearn.model_selection import KFold
from matplotlib.lines import Line2D


def plot_hist(folder,layer,Z_var_list,Z_note_list,Z_ontitle,Z_onlabel):
    datatype='Original'
    if folder=='Mode':
        name_corelist=['Bike','Car','Bus','Rideshare']
        var_label=['Bike','Car','Bus','Rideshare']

        FP=False
    if folder=='WFH_Group':
        name_corelist=['TB','GPI','WFH','WFHO']
        var_label=['Travel \n Burden','Gas Price \n Impact','WFH','WFH \n Option']
        FP=True

    if layer==3:
        model_path ='model_3lyr_'
    elif layer==0:
        model_path ='model_base_'


    for a in zip(Z_var_list,Z_note_list,Z_ontitle,Z_onlabel):
        Z_var=a[0]
        Z_note=a[1]
        protected_var=a[2]
        Z_list=a[3]
        base=False



        plt.style.use('seaborn-whitegrid')
        df_his=[]

        for k in zip(name_corelist):
            name_core=k[0]
            run_name =model_path+name_core

            df_temp=pd.read_csv('results/' + folder+'/'+ datatype+'/'+ a[1]+'/'+run_name+'.csv',index_col=None)
            with open('results/' + folder+'/'+ datatype+'/'+ a[1]+'/'+run_name+'.pkl', 'rb') as f:
                p_dic = pickle.load(f)
                FNR_z1=np.mean(np.array(p_dic['test_cm1'])[:,:,2]/(np.array(p_dic['test_cm1'])[:,:,2]+\
                                                            np.array(p_dic['test_cm1'])[:,:,3]))
                FNR_z0=np.mean(np.array(p_dic['test_cm0'])[:,:,2]/(np.array(p_dic['test_cm0'])[:,:,2]+\
                                                            np.array(p_dic['test_cm0'])[:,:,3]))
                FPR_z1=np.mean(np.array(p_dic['test_cm1'])[:,:,1]/(np.array(p_dic['test_cm1'])[:,:,0]+\
                                                            np.array(p_dic['test_cm1'])[:,:,1]))
                FPR_z0=np.mean(np.array(p_dic['test_cm0'])[:,:,1]/(np.array(p_dic['test_cm0'])[:,:,0]+\
                                                            np.array(p_dic['test_cm0'])[:,:,1]))
                accuracy_test=np.mean(np.array(p_dic['test_accuracy']))

            if FP:
                Z_list=[Z_list[0],Z_list[1],'Accuracy']
                df_his.append([FPR_z0,FPR_z1,accuracy_test])
            else:
                Z_list=[Z_list[0],Z_list[1],'Accuracy']
                df_his.append([FNR_z0,FNR_z1,accuracy_test])


        bar_df_base=pd.DataFrame(df_his,columns=Z_list,index=name_corelist)
        print('')
        print(Z_note)
        print(bar_df_base)


        bar_df_base=bar_df_base.round(3)

        # set width of bar
        barWidth = 0.25
        # x value & y value 
        min_bar=np.array(bar_df_base.loc[:,Z_list[0]])
        maj_bar=np.array(bar_df_base.loc[:,Z_list[1]])
        accuracy_score=np.array(1-bar_df_base['Accuracy'])
        print (accuracy_score)

        r1 = np.arange(len(min_bar))
        r2 = [x + barWidth for x in r1]

        fig, ax = plt.subplots(1, 1)
        bar1=ax.bar(r1, min_bar, color='#557f2d', width=barWidth, edgecolor='white', label=Z_list[0])
        bar2=ax.bar(r2, maj_bar, color='#7f6d5f', width=barWidth, edgecolor='white', label=Z_list[1])
        plt.xticks([r + 0.5*barWidth for r in range(len(min_bar))],var_label,size=14)
        plt.yticks(size=14)
        ax2 = ax.twinx()
        line = ax2.plot([r + 0.5*barWidth for r in range(len(min_bar))], accuracy_score, 'ks-', label = 'Total error rate',\
                        linewidth=0.5, markersize=6)
        plt.yticks(size=14)


        lns = [bar1]+[bar2]+line
        labs = [l.get_label() for l in lns]

        if folder=='Mode':  
            if Z_note=='Rural':
                plt.legend(lns, labs, fontsize=14,loc='center', bbox_to_anchor=(0.34, 0.8), frameon=True)
            elif Z_note=='Medcond':
                plt.legend(lns, labs, fontsize=14,loc='center', bbox_to_anchor=(0.5, 0.8), frameon=True)
            else:
                plt.legend(lns, labs, fontsize=14,loc='center right', bbox_to_anchor=(1, 0.8), frameon=True)

        if folder=='WFH_Group':
            plt.legend(lns, labs, fontsize=14,loc='center right', bbox_to_anchor=(1, 0.8), frameon=True)




        # Add xticks on the middle of the group bars
        ax.set_xlabel('Dependent Variable', fontweight='bold',size=18)
        if FP:
            ax.set_ylabel('False Positive Rate', fontweight='bold',size=18)
        else:
            ax.set_ylabel('False Negative Rate', fontweight='bold',size=18)
        ax2.set_ylabel('Total Error Rate', fontweight='bold',size=18)
        if layer==3:
            ax.set_ylim((0,0.3))
            ax2.set_ylim((0,0.3))
            

        ax.set_ylim((0,1))
        ax2.set_ylim((0,1))
            


        if FP:
            plt.title('False Positive Rate by '+protected_var, fontweight='bold',size=20,loc='center')
            plt.savefig('plots/' +  folder+'/'+ model_path+'/initial/'+Z_note+'.png', bbox_inches="tight", dpi=500)
        else:
            plt.title('False Negative Rate by '+protected_var, fontweight='bold',size=20,loc='center')
            plt.savefig('plots/' +  folder+'/'+ model_path+'/initial/'+Z_note+'.png', bbox_inches="tight", dpi=500)
