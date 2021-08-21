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
from tqdm.notebook import tqdm

def correlation(x, y, w): 
        eps=1e-20
        mx=tf.reduce_sum(w*x)/tf.reduce_sum(w)
        my=tf.reduce_sum(w*y)/tf.reduce_sum(w)
        xm, ym = x-mx, y-my
        r_num = tf.reduce_sum(w*tf.multiply(xm,ym))        
        r_den =(tf.sqrt(tf.reduce_sum(w*tf.square(xm)))+eps)*(tf.sqrt(tf.reduce_sum(w*tf.square(ym)))+eps)
        return r_num / r_den

class FeedForward_DNN:
    def __init__(self,K,MODEL_NAME,INCLUDE_VAL_SET,INCLUDE_RAW_SET, RUN_DIR):
        self.graph = tf.Graph()
        self.K = K
        self.MODEL_NAME = MODEL_NAME
        self.INCLUDE_VAL_SET = INCLUDE_VAL_SET
        self.INCLUDE_RAW_SET=INCLUDE_RAW_SET
        self.RUN_DIR = RUN_DIR

    
    def init_hyperparameter(self, lam,lr,lyr,n_epoch,n_mini_batch):
        # h stands for hyperparameter
        self.h = {}

        self.h['M']=lyr
        self.h['M_2']=2
        self.h['n_hidden']=200
        self.h['l1_const']=1e-5
        self.h['l2_const']=0.001
        self.h['dropout_rate']=0.01
 
        self.h['batch_normalization']=True
        self.h['learning_rate']=lr
        self.h['n_iteration']=5000
        self.h['n_mini_batch']=n_mini_batch
        
        self.h['lam']=lam
        self.h['n_epoch']=n_epoch

    def change_hyperparameter(self, new_hyperparameter):
        assert bool(self.h) == True
        self.h = new_hyperparameter
    
    def random_sample_hyperparameter(self):
        assert bool(self.hs) == True
        assert bool(self.h) == True
        for name_ in self.h.keys():
            self.h[name_] = np.random.choice(self.hs[name_+'_list'])

    def obtain_mini_batch(self,i,full=True):

        index=np.arange((i-1)*self.h['n_mini_batch'],i*self.h['n_mini_batch'])
        if full==False:
            index=np.arange((i-1)*self.h['n_mini_batch'],self.N_train)
        self.X_batch = self.X_train_[index, :]
        self.Y_batch = self.Y_train_[index]
        self.Z_batch = self.Z_train_[index]
        self.W_batch = self.W_train_[index]

        
    def load_data(self, Z_var,input_data_raw = None):
        print("Loading datasets...")
        Z_train= X_train[Z_var]
        Z_test = X_test[Z_var]
        self.colnames = list(X_train.columns)

        self.X_train = X_train.values
        self.Y_train = Y_train.values
        self.Z_train = Z_train.values

        self.X_test=X_test.values
        self.Y_test=Y_test.values
        self.Z_test=Z_test.values
    
        
        self.X_z0_train=X_train[Z_train==1].values
        self.X_z1_train=X_train[Z_train==0].values

        self.Y_z0_train=Y_train[Z_train==1].values
        self.Y_z1_train=Y_train[Z_train==0].values
        
        self.X_z0_test=X_test[Z_test==1].values
        self.X_z1_test=X_test[Z_test==0].values
        self.Y_z0_test=Y_test[Z_test==1].values
        self.Y_z1_test=Y_test[Z_test==0].values
        
        self.W_z0_train=W_train[Z_train==1].values
        self.W_z1_train=W_train[Z_train==0].values
        self.W_z0_test=W_test[Z_test==1].values
        self.W_z1_test=W_test[Z_test==0].values
    
        if self.INCLUDE_VAL_SET:
            self.X_val = X_val.values
            self.Y_val = Y_val.values
                
        print("Training set", self.X_train.shape, self.Y_train.shape, self.Z_train.shape)
        print("Testing set", self.X_test.shape, self.Y_test.shape, self.Z_test.shape)
        if self.INCLUDE_VAL_SET:
            print("Validation set", self.X_val.shape, self.Y_val.shape)
            
        self.W_train = W_train.values
        self.W_test = W_test.values

        # save dim
        self.N_train,self.D = self.X_train.shape
        self.N_test,self.D = self.X_test.shape
        

    def bootstrap_data(self, N_bootstrap_sample):

        self.N_bootstrap_sample = N_bootstrap_sample
        bootstrap_sample_index = np.random.choice(self.N_train, size = self.N_bootstrap_sample) 
        self.X_train_ = self.X_train
        self.Y_train_ = self.Y_train
        self.Z_train_ = self.Z_train
        self.W_train_ = self.W_train
        #save positive y index
        

    def standard_hidden_layer(self, name):
        # standard layer, repeated in the following for loop.
        self.hidden = tf.layers.dense(self.hidden, self.h['n_hidden'], activation = tf.nn.relu, name = name)
        if self.h['batch_normalization'] == True:
            self.hidden = tf.layers.batch_normalization(inputs = self.hidden, axis = 1)
        self.hidden = tf.layers.dropout(inputs = self.hidden, rate = self.h['dropout_rate'])
        
    

    def build_model(self, method):

        with self.graph.as_default():
            self.X = tf.placeholder(dtype = tf.float32, shape = (None, self.D), name = 'X')
            self.Y = tf.placeholder(dtype = tf.int64, shape = (None), name = 'Y')
            self.Z = tf.placeholder(dtype = tf.int64, shape = (None), name = 'Z')
            self.W = tf.placeholder(dtype=tf.float32, shape=(None),name='W')
            
            self.hidden = self.X
            
            for i in range(self.h['M']):
                name = 'hidden'+str(i)
                self.standard_hidden_layer(name)
            # last layer: utility in choice models
            self.output=tf.layers.dense(self.hidden, self.K, name = 'output')
            self.prob=tf.nn.softmax(self.output, name = 'prob')
            self.cl=tf.argmax(self.prob,1,name='class')
            self.output_tensor = tf.identity(self.output, name='logits')
                        

            l1_l2_regularization = tf.contrib.layers.l1_l2_regularizer(scale_l1=self.h['l1_const'], scale_l2=self.h['l2_const'], scope=None)
            vars_ = tf.trainable_variables()
            weights = [var_ for var_ in vars_ if 'kernel' in var_.name]
            regularization_penalty = tf.contrib.layers.apply_regularization(l1_l2_regularization, vars_)
            
            # evaluate
            self.correct = tf.equal(self.cl, self.Y, name='correct')
            self.accuracy,self.update_op = tf.metrics.accuracy(labels=self.Y, predictions=self.cl,weights=self.W,name='accuracy')
            self.confusion_matrix = tf.confusion_matrix(self.Y,self.cl,weights=self.W,name='confusion_matrix')
            
            # Isolate the variables stored behind the scenes by the metric operation
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy")
            
            # Define initializer to initialize/reset running variables
            self.running_vars_initializer = tf.variables_initializer(var_list=running_vars)

            # loss function

            self.cost1 = (1-self.h['lam'])*tf.reduce_sum(self.W*tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.output, labels = self.Y), name = 'cost1')/tf.reduce_sum(self.W)

            iy=tf.transpose(tf.where(self.Y>0))
            iy2=tf.transpose(tf.where(self.Y<1))
    
            self.z2=tf.gather(self.Z, iy)
            self.c2=tf.gather(self.cl, iy)
            
            self.cr=correlation(tf.cast(tf.gather(self.Z, iy), 'float'), tf.cast(tf.gather(self.cl,iy), 'float'), tf.cast(tf.gather(self.W,iy), 'float'))
            if method=='cor_soft':
                self.prob_one=tf.squeeze(tf.gather(self.prob, [1], axis=1))
                self.cost2 =self.h['lam']*abs(correlation(tf.cast(tf.gather(self.Z, iy), 'float'), tf.cast(tf.gather(self.prob_one,iy), 'float'), tf.cast(tf.gather(self.W,iy), 'float')))
                self.cost=self.cost2+self.cost1
                
            if method=='cor_soft_FP':
                self.prob_one=tf.squeeze(tf.gather(self.prob, [1], axis=1))
                self.cost2 =self.h['lam']*abs(correlation(tf.cast(tf.gather(self.Z, iy2), 'float'), tf.cast(tf.gather(self.prob_one,iy2), 'float'), tf.cast(tf.gather(self.W,iy2), 'float')))
                self.cost=self.cost2+self.cost1

            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.h['learning_rate']) # opt objective
            self.training_op = self.optimizer.minimize(self.cost) # minimize the opt objective
            self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(),self.running_vars_initializer)
            self.saver= tf.train.Saver()
                        
    def train_model(self):
        self.train_accuracy_list=[]
        self.test_accuracy_list=[]
        self.train_cost1_list=[]
        self.test_cost1_list=[]
        self.train_cost2_list=[]
        self.test_cost2_list=[]
        
        with tf.Session(graph=self.graph) as sess:
            self.init.run()
            
            #for early stopping:
            best_train_accuracy=0
            best_cost=100
            stop=False
            last_improvement=0
            require_improvement=20

            i=0

            for i in tqdm(range(self.h['n_epoch'])):

                for k in range(1,self.N_train//self.h['n_mini_batch']+1):
                    sess.run(self.running_vars_initializer)
                    # gradient descent
                    self.obtain_mini_batch(k)
                    sess.run(self.training_op, feed_dict = {self.X: self.X_batch, self.Y: self.Y_batch, self.Z: self.Z_batch, self.W: self.W_batch})


                for k in range(self.N_train//self.h['n_mini_batch']+1,self.N_train//self.h['n_mini_batch']+2):
                    sess.run(self.running_vars_initializer)
                    self.obtain_mini_batch(k,full=False)
                    sess.run(self.training_op, feed_dict = {self.X: self.X_batch, self.Y: self.Y_batch, self.Z: self.Z_batch, self.W: self.W_batch})
                self.cl_train =sess.run(self.cl,feed_dict={self.X: self.X_train_, self.Y: self.Y_train_, self.Z: self.Z_train_})
                self.cl_test =sess.run(self.cl,feed_dict={self.X: self.X_test, self.Y: self.Y_test, self.Z: self.Z_test})
                
                self.update_train_temp = sess.run(self.update_op,feed_dict = {self.cl: self.cl_train, self.Y: self.Y_train_, self.W: self.W_train_})
                self.accuracy_train_temp = sess.run(self.accuracy)
                self.update_test_temp = sess.run(self.update_op,feed_dict = {self.cl: self.cl_test, self.Y: self.Y_test, self.W: self.W_test})
                self.accuracy_test_temp = sess.run(self.accuracy)
            
                self.train_accuracy_list.append(self.accuracy_train_temp)
                self.test_accuracy_list.append(self.accuracy_test_temp)
            
                
                

                if self.h['lam']!=0:
                    self.cost1_train_temp = self.cost1.eval(feed_dict = {self.X: self.X_train_, self.Y: self.Y_train_, self.Z: self.Z_train_, self.W: self.W_train_})
                    self.cost1_test_temp = self.cost1.eval(feed_dict = {self.X: self.X_test, self.Y: self.Y_test, self.Z: self.Z_test, self.W: self.W_test})
                    self.cost2_train_temp = self.cost2.eval(feed_dict = {self.X: self.X_train_, self.Y: self.Y_train_, self.Z: self.Z_train_, self.W: self.W_train_})
                    self.cost2_test_temp = self.cost2.eval(feed_dict = {self.X: self.X_test, self.Y: self.Y_test, self.Z: self.Z_test, self.W: self.W_test})

                    self.train_cost1_list.append(self.cost1_train_temp)
                    self.test_cost1_list.append(self.cost1_test_temp)
                    self.train_cost2_list.append(self.cost2_train_temp)
                    self.test_cost2_list.append(self.cost2_test_temp)

                    self.confusion_matrix_z0_train =self.confusion_matrix.eval(feed_dict={self.X: self.X_z0_train, self.Y: self.Y_z0_train, self.Z: self.Z_train, self.W: self.W_z0_train})
                    self.confusion_matrix_z1_train =self.confusion_matrix.eval(feed_dict={self.X: self.X_z1_train, self.Y: self.Y_z1_train, self.Z: self.Z_train, self.W: self.W_z1_train})

                    self.confusion_matrix_z0_test =self.confusion_matrix.eval(feed_dict={self.X: self.X_z0_test, self.Y: self.Y_z0_test, self.Z: self.Z_test, self.W: self.W_z0_test})
                    self.confusion_matrix_z1_test =self.confusion_matrix.eval(feed_dict={self.X: self.X_z1_test, self.Y: self.Y_z1_test, self.Z: self.Z_test, self.W: self.W_z1_test})

                    train_cm0=self.confusion_matrix_z0_train.ravel()
                    train_cm1=self.confusion_matrix_z1_train.ravel()
                    test_cm0=self.confusion_matrix_z0_test.ravel()
                    test_cm1=self.confusion_matrix_z1_test.ravel()
   

                    train_FNR0=train_cm0[2]/(train_cm0[2]+train_cm0[3])
                    train_FNR1=train_cm1[2]/(train_cm1[2]+train_cm1[3])
                    train_FNR_gap=train_FNR0-train_FNR1


                    test_FNR0=test_cm0[2]/(test_cm0[2]+test_cm0[3])
                    test_FNR1=test_cm1[2]/(test_cm1[2]+test_cm1[3])
                    test_FNR_gap=test_FNR0-test_FNR1
                    
                    train_FPR0=train_cm0[1]/(train_cm0[0]+train_cm0[1])
                    train_FPR1=train_cm1[1]/(train_cm1[0]+train_cm1[1])
                    train_FPR_gap=train_FPR0-train_FPR1

                    test_FPR0=test_cm0[1]/(test_cm0[0]+test_cm0[1])
                    test_FPR1=test_cm1[1]/(test_cm1[0]+test_cm1[1])
                    test_FPR_gap=test_FPR0-test_FPR1

                    if i%100==0:
                        print("Epoch ", i," Accuracy_train = ", self.accuracy_train_temp,\
                             " Cost1_train = ",self.cost1_train_temp,\
                             " Cost2_train = ",self.cost2_train_temp,\
                             " Cost_train = ",self.cost1_train_temp+self.cost2_train_temp)

                        if not FP:
                            print(" FNR_gap_train = ", train_FNR_gap,\
                                 " FNR_gap_test = ",test_FNR_gap)
                        else:
                            print(" FPR_gap_train = ", train_FPR_gap,\
                                 " FPR_gap_test = ",test_FPR_gap)
                            print(" FNR_gap_train = ", train_FNR_gap,\
                                 " FNR_gap_test = ",test_FNR_gap)

                    if best_cost > self.cost1_train_temp+self.cost2_train_temp:
                        save_sess=sess
                        best_cost=self.cost1_train_temp+self.cost2_train_temp
                        last_improvement = 0
                        self.best_iter=i
                        self.saver.save(sess, self.RUN_DIR+self.MODEL_NAME+".ckpt")
                    else:
                        last_improvement +=1

#                     if last_improvement > require_improvement:
#                         print("No improvement found during the "+str(require_improvement)+" last iterations, stopping optimization.")
#                         # Break out from the loop.
#                         stop = True

                else:
                    if i%100==0:
                        self.confusion_matrix_z0_train =self.confusion_matrix.eval(feed_dict={self.X: self.X_z0_train, self.Y: self.Y_z0_train, self.Z: self.Z_train, self.W: self.W_z0_train})
                        self.confusion_matrix_z1_train =self.confusion_matrix.eval(feed_dict={self.X: self.X_z1_train, self.Y: self.Y_z1_train, self.Z: self.Z_train, self.W: self.W_z1_train})

                        self.confusion_matrix_z0_test =self.confusion_matrix.eval(feed_dict={self.X: self.X_z0_test, self.Y: self.Y_z0_test, self.Z: self.Z_test, self.W: self.W_z0_test})
                        self.confusion_matrix_z1_test =self.confusion_matrix.eval(feed_dict={self.X: self.X_z1_test, self.Y: self.Y_z1_test, self.Z: self.Z_test, self.W: self.W_z1_test})

                        train_cm0=self.confusion_matrix_z0_train.ravel()
                        train_cm1=self.confusion_matrix_z1_train.ravel()
                        test_cm0=self.confusion_matrix_z0_test.ravel()
                        test_cm1=self.confusion_matrix_z1_test.ravel()

                        train_FNR0=train_cm0[2]/(train_cm0[2]+train_cm0[3])
                        train_FNR1=train_cm1[2]/(train_cm1[2]+train_cm1[3])
                        train_FNR_gap=train_FNR0-train_FNR1


                        test_FNR0=test_cm0[2]/(test_cm0[2]+test_cm0[3])
                        test_FNR1=test_cm1[2]/(test_cm1[2]+test_cm1[3])
                        test_FNR_gap=test_FNR0-test_FNR1
                        
                        train_FPR0=train_cm0[1]/(train_cm0[0]+train_cm0[1])
                        train_FPR1=train_cm1[1]/(train_cm1[0]+train_cm1[1])
                        train_FPR_gap=train_FPR0-train_FPR1

                        test_FPR0=test_cm0[1]/(test_cm0[0]+test_cm0[1])
                        test_FPR1=test_cm1[1]/(test_cm1[0]+test_cm1[1])
                        test_FPR_gap=test_FPR0-test_FPR1
                        
                        print("Epoch ", i," Accuracy_train = ", self.accuracy_train_temp,\
                                 " Accuracy_test = ",self.accuracy_test_temp)
                        if not FP:
                            print(" FNR_gap_train = ", train_FNR_gap,\
                                 " FNR_gap_test = ",test_FNR_gap)
                        else:
                            print(" FPR_gap_train = ", train_FPR_gap,\
                                 " FPR_gap_test = ",test_FPR_gap)
                            print(" FNR_gap_train = ", train_FNR_gap,\
                                 " FNR_gap_test = ",test_FNR_gap)
    
                    if best_train_accuracy < self.accuracy_train_temp:
                        save_sess=sess
                        best_train_accuracy=self.accuracy_train_temp
                        last_improvement = 0
                        self.best_iter=i
                        self.saver.save(sess, self.RUN_DIR+self.MODEL_NAME+".ckpt")
                    else:
                        last_improvement +=1

#                     if last_improvement > require_improvement:
#                         print("No improvement found during the "+str(require_improvement)+" last iterations, stopping optimization.")
#                         # Break out from the loop.
#                         stop = True

                self.final_iter=i                                  
                i+=1
                
    def evaluate_model(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            
            saver = tf.train.import_meta_graph(self.RUN_DIR+self.MODEL_NAME+ ".ckpt.meta")
            saver.restore(sess, self.RUN_DIR+self.MODEL_NAME+ ".ckpt")
            graph = tf.get_default_graph()
            
            self.X = graph.get_tensor_by_name("X:0")
            self.Y = graph.get_tensor_by_name("Y:0")
            self.Z = graph.get_tensor_by_name("Z:0")
            self.W = graph.get_tensor_by_name("W:0")
            
            self.cl= graph.get_tensor_by_name("class:0")

            self.confusion_matrix = tf.confusion_matrix(self.Y,self.cl,weights=self.W, name='confusion_matrix')
            self.accuracy = tf.metrics.accuracy(labels=self.Y, predictions=self.cl,weights=self.W,name='accuracy')[1]

            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy")
            running_vars_initializer = tf.variables_initializer(var_list=running_vars)
            
            self.init=tf.group(tf.local_variables_initializer())
            
            self.init.run()
            
            self.cl_z0_train =sess.run(self.cl,feed_dict={self.X: self.X_z0_train, self.Y: self.Y_z0_train, self.Z: self.Z_train})
            self.cl_z1_train =sess.run(self.cl,feed_dict={self.X: self.X_z1_train, self.Y: self.Y_z1_train, self.Z: self.Z_train})

            self.cl_z0_test =sess.run(self.cl,feed_dict={self.X: self.X_z0_test, self.Y: self.Y_z0_test, self.Z: self.Z_test})
            self.cl_z1_test =sess.run(self.cl,feed_dict={self.X: self.X_z1_test, self.Y: self.Y_z1_test, self.Z: self.Z_test})

            self.cl_train =sess.run(self.cl,feed_dict={self.X: self.X_train_, self.Y: self.Y_train_, self.Z: self.Z_train_})
            self.cl_test =sess.run(self.cl,feed_dict={self.X: self.X_test, self.Y: self.Y_test, self.Z: self.Z_test})
                    
            self.accuracy_train = self.accuracy.eval(feed_dict={self.Y: self.Y_train_, self.cl:self.cl_train,self.W: self.W_train_})
            self.accuracy_test = self.accuracy.eval(feed_dict={self.Y: self.Y_test, self.cl:self.cl_test,self.W: self.W_test})
        
            self.confusion_matrix_z0_train =self.confusion_matrix.eval(feed_dict={self.Y: self.Y_z0_train, self.cl: self.cl_z0_train, self.W: self.W_z0_train})
            self.confusion_matrix_z1_train =self.confusion_matrix.eval(feed_dict={self.Y: self.Y_z1_train, self.cl: self.cl_z1_train, self.W: self.W_z1_train})

            self.confusion_matrix_z0_test =self.confusion_matrix.eval(feed_dict={self.Y: self.Y_z0_test, self.cl:self.cl_z0_test, self.W: self.W_z0_test})
            self.confusion_matrix_z1_test =self.confusion_matrix.eval(feed_dict={self.Y: self.Y_z1_test, self.cl:self.cl_z1_test, self.W: self.W_z1_test})            
