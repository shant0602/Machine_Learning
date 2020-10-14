import numpy as np
import math
import pandas as pd
import math
# from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
# import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
# import xgboost
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.metrics import roc_curve

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score,\
    precision_score, recall_score, confusion_matrix, plot_confusion_matrix,\
    precision_recall_curve, plot_precision_recall_curve, average_precision_score
import itertools
import timeit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics



import mlrose_hiive as mlrose
import numpy as np
import concurrent.futures
import pandas as pd
from sklearn.datasets import load_breast_cancer
from mlrose_hiive.runners._runner_base import _RunnerBase
import mlrose_hiive.runners.sa_runner as sa
import mlrose_hiive.runners.rhc_runner as rhc
import mlrose_hiive.runners.ga_runner as ga
import mlrose_hiive.runners.mimic_runner as mimic
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import time
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from numba import jit, cuda 
import logging
from sklearn.model_selection import GridSearchCV, cross_val_score
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 
np.random.seed(3341)
problem_size = 25


class optimization_problem:
    def __init__(self, name):
        print(name + ' optimization')
        self.noOfiteration=10000
    # @jit(target ="cuda")  
    def hypertuning_rscv(self,list_of_tup):
        estimator, p_distr, nbr_iter, X, y, cval=list_of_tup
        print('***************************************************Started RDSearch*******************************************************************************')
        rdmsearch = RandomizedSearchCV(
        estimator,
        param_distributions=p_distr,
        n_jobs=-1,
        n_iter=nbr_iter,
        cv=cval,
        random_state=1, scoring='accuracy',verbose=10,refit=True
    )
        # CV = Cross-Validation ( here using Stratified KFold CV)
        clf = rdmsearch.fit(X, y)
        # print('RANDOM CV RESULTs',clf.cv_results_)

        return clf.best_params_, clf.best_score_, clf.best_estimator_,clf.cv_results_


#*******************************************************************************************TSP*****************************************************************************************************

class NN(optimization_problem):
    def __init__(self, size):
        super().__init__('NN')
        self.problem_size = size
        # self.noOfiteration=10000
        self.noOfiteration=5000


    def sa(self):
        iteration = self.noOfiteration
        problem_size_space=self.problem_size
        step=problem_size_space//2
        # temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
        temperature_list=[1, 10, 50]
        # sa_params = { 'hidden_nodes': [(3,), (4,), (5,), (5, 5)],
        sa_params ={'hidden_nodes':  [(3,),(5,),(5, 5),(10, 10, 10),(5, 5, 5)],
                                         #(10, 10), (5, 5, 5), (10, 10, 10), (20, 20, 20)]
                        'max_iters': [x for x in range(0,iteration+1,1000)],
                        'schedule':[ mlrose.GeomDecay(init_temp=init_temp) for init_temp in temperature_list],
                        'learning_rate':[0.001,0.01,0.1],
                        'activation': ['tanh', 'relu','sigmoid']

                    }
                    
        sa_model = mlrose.NeuralNetwork(random_state=1, 
                                        algorithm ='simulated_annealing', 
                                        bias = False, is_classifier = True, 
                                        learning_rate=0.001, early_stopping = True, 
                                        clip_max = 5, max_attempts = 5000,curve=True)
        return sa_model,sa_params

    def rhc(self):
        iteration = self.noOfiteration
        problem_size_space=self.problem_size
        step=problem_size_space//2
        # rhc_params = { 'hidden_nodes': [(3,), (4,), (5,), (5, 5)],
        rhc_params =   {'hidden_nodes':  [(3,),(5,),(5, 5),(10, 10, 10),(5, 5, 5)],
                                         #(10, 10), (5, 5, 5), (10, 10, 10), (20, 20, 20)]
                        'max_iters': [x for x in range(0,iteration+1,1000)],
                        'restarts': [x for x in range(0,problem_size_space+1,step)],
                        'activation': ['tanh', 'relu','sigmoid'],
                        'learning_rate':[0.001,0.01,0.1]
                    }
                    
        rhc_model = mlrose.NeuralNetwork(random_state=1, 
                                        algorithm ='random_hill_climb', 
                                        bias = False, is_classifier = True, 
                                        learning_rate=0.001, early_stopping = True, 
                                        clip_max = 5, max_attempts = 5000,curve=True)
        return rhc_model,rhc_params

    def ga(self):
        iteration = self.noOfiteration
        problem_size_space=self.problem_size
        problem_size_space_exp=problem_size_space*10
        step=problem_size_space_exp//2
        # ga_params = { 'hidden_nodes': [(3,), (4,), (5,), (5, 5)],
        ga_params ={'hidden_nodes': [(3,),(5, 5),(10, 10, 10)],
                                         #(10, 10), (5, 5, 5), (10, 10, 10), (20, 20, 20)]
                        'max_iters': [x for x in range(0,iteration+1,1000)],
                        'pop_size':[x for x in range(step,problem_size_space_exp+1,step)],
                        'mutation_prob':np.arange(.1,.7,.2),
                        'learning_rate':[0.001,0.01,0.1]
                    }
                    
        ga_model = mlrose.NeuralNetwork(random_state=1, 
                                        algorithm ='genetic_alg', 
                                        bias = False, is_classifier = True, 
                                        learning_rate=0.001, early_stopping = True, 
                                        clip_max = 5, max_attempts = 5000,curve=True)
        return ga_model,ga_params

    def backprop(self):
        iteration = self.noOfiteration
        problem_size_space=self.problem_size
        problem_size_space_exp=problem_size_space*10
        step=problem_size_space_exp//2
        # backprop_params = { 'hidden_nodes': [(3,), (4,), (5,), (5, 5)],
        backprop_params ={'hidden_nodes': [(3,),(5, 5),(10, 10, 10),(5, 5, 5), (10, 10, 10),  (20,20,20),(5,)],
                                         #(10, 10), (5, 5, 5), (10, 10, 10), (20, 20, 20)]
                        'activation': ['tanh', 'relu','sigmoid'],
                        'max_iters': [x for x in range(0,iteration+1,1000)],
                        'learning_rate':[0.001,0.005,0.01,0.1,0.5]

                    }
                    
        backprop_model = mlrose.NeuralNetwork(random_state=1, 
                                        algorithm ='gradient_descent', 
                                        bias = False, is_classifier = True, 
                                         early_stopping = True, 
                                        clip_max = 5, max_attempts = 5000,curve=True)
        return backprop_model,backprop_params

    def optimize(self):
        X, y, X_train, X_test, y_train, y_test = self.get_data()
        mod,param=[],[]
        # model,param_grid=self.sa()
        # mod.append(model)
        # param.append(param_grid)
        # model,param_grid=self.rhc()
        # mod.append(model)
        # param.append(param_grid)
        # model,param_grid=self.ga()
        # mod.append(model)
        # param.append(param_grid)
        model,param_grid=self.backprop()
        mod.append(model)
        param.append(param_grid)
        # model,param_grid=self.ga()
        # mod.append(model)
        # param.append(param_grid)
        args=[(mod[i],param[i],40,X,y,10) for i in range(len(mod))]
        i=3
        dic={1:'SA_',2:'RHC_',3:'backprop_2_',4:'GA'}
        for tup in args:
            rf_parameters, rf_ht_score, classifier_bestFit,res=self.hypertuning_rscv(tup)
            reg_table = pd.DataFrame(res)
            # grid_table = pd.DataFrame(scores)
            reg_table.to_csv('NN/{}_reg.csv'.format(dic[i]), index=False)
            # grid_table.to_csv('{}_grid.csv'.format(str(i)), index=False)
            print('bestParam:',rf_parameters, rf_ht_score)
            i+=1

        # with concurrent.futures.ProcessPoolExecutor() as f:
        #     results = f.map(self.hypertuning_rscv,args)
        # for res in results:
        #     rf_parameters, rf_ht_score, classifier_bestFit=res
        #     print('bestParam:',rf_parameters, rf_ht_score)

    def evaluate_sa(self,X, y, X_train, X_test, y_train, y_test):
        acc_train=[]
        acc_test=[]

        sa_model = mlrose.NeuralNetwork(random_state=1, 
                                algorithm ='simulated_annealing', 
                                bias = False, is_classifier = True, 
                                learning_rate=0.1, early_stopping = True, 
                                clip_max = 5, max_attempts = 4000,curve=True,\
                                    hidden_nodes=(5,5),schedule=mlrose.GeomDecay(init_temp=1),\
                                        max_iters=9000,activation='relu')
        percent=0.80
        while percent >=0.20:
            X_train2, X_test2, y_train2, y_test2 = train_test_split(
                X_train, y_train, random_state=0, test_size=percent)
            model_object=sa_model.fit(X_train2,y_train2)
            # y_train_pred=sa_model.predict(X_train2)
            # y_train_accuracy = accuracy_score(y_train2, y_train_pred)
            y_train_accuracy = cross_val_score(model_object,X_train2,y_train2,scoring='accuracy',cv=10,n_jobs=-1).mean()

            acc_train.append(y_train_accuracy)
            print('SA train acc with model trained on {}% of training data='.format(str(math.ceil((1-percent)*100))),y_train_accuracy)
            y_test_pred = model_object.predict(X_test)
            # y_test_accuracy=cross_val_score(model_object,X_test,scoring='accuracy',cv=10,n_jobs=-1).mean()
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            acc_test.append(y_test_accuracy)
            print('SA test acc with model trained on {}% of training data='.format(str(math.ceil((1-percent)*100))),y_test_accuracy)
            # loss_sa=pd.DataFrame({'Loss_value':model_object.fitness_curve})
            # loss_sa.to_csv('NN/sa'+'_loss_curve.csv')
            # print(model_object.fitness_curve)

            percent-=0.2 
        model_object=sa_model.fit(X_train,y_train)
        # y_train_pred=sa_model.predict(X_train)
        # y_train_accuracy = accuracy_score(y_train, y_train_pred)
        start=time.perf_counter()
        y_train_accuracy = cross_val_score(model_object,X_train2,y_train2,scoring='accuracy',cv=10,n_jobs=-1).mean()
        end_time=time.perf_counter()
        print('SA time= {} cpu time units*******************************************************************'.format(end_time-start))
        acc_train.append(y_train_accuracy)
        print('SA train accacc with model trained on {}% of training data='.format(str(100)),y_train_accuracy)
        y_test_pred = model_object.predict(X_test)
        # y_test_accuracy=cross_val_score(model_object,X_test,scoring='accuracy',cv=10,n_jobs=-1).mean()
        y_test_accuracy = accuracy_score(y_test, y_test_pred)
        acc_test.append(y_test_accuracy)
        print('SA test acc with model trained on {}% of training data='.format(str(100)),y_test_accuracy)
        loss_sa=pd.DataFrame({'Loss_value':model_object.fitness_curve})
        loss_sa.to_csv('NN/sa'+'_loss_curve.csv')
        lc_sa=pd.DataFrame({'Train Acuuracy':acc_train,'Test Accuracy':acc_test,'% of training data':[20,40,60,80,100]})
        lc_sa.to_csv('NN/sa'+'_learning_curve.csv')
        # print(model_object.fitness_curve)
        # print(model_object.curve)
        print('*******************************************************************************************************************')
         

    def evaluate_rhc(self,X, y, X_train, X_test, y_train, y_test):
        acc_train=[]
        acc_test=[]
        rhc_model = mlrose.NeuralNetwork(random_state=1, 
                                        algorithm ='random_hill_climb', 
                                        bias = False, is_classifier = True, 
                                        learning_rate=0.1, early_stopping = True, 
                                        clip_max = 5, max_attempts = 5000,curve=True,restarts=2,
                                        max_iters=5000,hidden_nodes=(5,5),activation='sigmoid')
        percent=0.80
        while percent >=0.20:
            X_train2, X_test2, y_train2, y_test2 = train_test_split(
                X_train, y_train, random_state=0, test_size=percent)
            model_object=rhc_model.fit(X_train2,y_train2)
            # y_train_pred=rhc_model.predict(X_train2)
            # y_train_accuracy = accuracy_score(y_train2, y_train_pred)
            y_train_accuracy = cross_val_score(model_object,X_train2,y_train2,scoring='accuracy',cv=10,n_jobs=-1).mean()
            acc_train.append(y_train_accuracy)
            print('rhc train acc with model trained on {}% of training data='.format(str(math.ceil((1-percent)*100))),y_train_accuracy)
            y_test_pred = model_object.predict(X_test)
            # y_test_accuracy=cross_val_score(model_object,X_test,scoring='accuracy',cv=10,n_jobs=-1).mean()
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            acc_test.append(y_test_accuracy)
            print('rhc test acc with model trained on {}% of training data='.format(str(math.ceil((1-percent)*100))),y_test_accuracy)
            # loss_rhc=pd.DataFrame({'Loss_value':model_object.fitness_curve})
            # loss_rhc.to_csv('NN/rhc'+'_loss_curve.csv')
            # print(model_object.fitness_curve)

            percent-=0.2 
        model_object=rhc_model.fit(X_train,y_train)
        # y_train_pred=rhc_model.predict(X_train)
        # y_train_accuracy = accuracy_score(y_train, y_train_pred)
        start=time.perf_counter()
        y_train_accuracy = cross_val_score(model_object,X_train2,y_train2,scoring='accuracy',cv=10,n_jobs=-1).mean()
        end_time=time.perf_counter()
        print('RHC time= {} cpu time units*********************************************************************************'.format(end_time-start))
        acc_train.append(y_train_accuracy)
        print('rhc train accacc with model trained on {}% of training data='.format(str(100)),y_train_accuracy)
        y_test_pred = model_object.predict(X_test)
        # y_test_accuracy=cross_val_score(model_object,X_test,scoring='accuracy',cv=10,n_jobs=-1).mean()
        y_test_accuracy = accuracy_score(y_test, y_test_pred)
        acc_test.append(y_test_accuracy)
        print('rhc test acc with model trained on {}% of training data='.format(str(100)),y_test_accuracy)
        loss_rhc=pd.DataFrame({'Loss_value':model_object.fitness_curve})
        loss_rhc.to_csv('NN/rhc'+'_loss_curve.csv')
        lc_rhc=pd.DataFrame({'Train Acuuracy':acc_train,'Test Accuracy':acc_test,'% of training data':[20,40,60,80,100]})
        lc_rhc.to_csv('NN/rhc'+'_learning_curve.csv')
        # print(model_object.fitness_curve)
        # print(model_object.curve)
        print('*******************************************************************************************************************')

    def evaluate_ga(self,X, y, X_train, X_test, y_train, y_test):
        acc_train=[]
        acc_test=[]
        ga_model = mlrose.NeuralNetwork(random_state=1, 
                                        algorithm ='genetic_alg', 
                                        bias = False, is_classifier = True, 
                                        early_stopping = True, 
                                        clip_max = 5, max_attempts = 5000,curve=True,pop_size=25,\
                                            mutation_prob=0.1,max_iters=1000,learning_rate=0.001,\
                                                hidden_nodes=(3,))
        percent=0.80
        while percent >=0.20:
            X_train2, X_test2, y_train2, y_test2 = train_test_split(
                X_train, y_train, random_state=0, test_size=percent)
            model_object=ga_model.fit(X_train2,y_train2)
            # y_train_pred=ga_model.predict(X_train2)
            # y_train_accuracy = accuracy_score(y_train2, y_train_pred)
            y_train_accuracy = cross_val_score(model_object,X_train2,y_train2,scoring='accuracy',cv=10,n_jobs=-1).mean()

            acc_train.append(y_train_accuracy)
            print('ga train acc with model trained on {}% of training data='.format(str(math.ceil((1-percent)*100))),y_train_accuracy)
            y_test_pred = model_object.predict(X_test)
            # y_test_accuracy=cross_val_score(model_object,X_test,scoring='accuracy',cv=10,n_jobs=-1).mean()
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            acc_test.append(y_test_accuracy)
            print('ga test acc with model trained on {}% of training data='.format(str(math.ceil((1-percent)*100))),y_test_accuracy)
            # loss_ga=pd.DataFrame({'Loss_value':model_object.fitness_curve})
            # loss_ga.to_csv('NN/ga'+'_loss_curve.csv')
            # print(model_object.fitness_curve)

            percent-=0.2 
        model_object=ga_model.fit(X_train,y_train)
        # y_train_pred=ga_model.predict(X_train)
        # y_train_accuracy = accuracy_score(y_train, y_train_pred)
        start=time.perf_counter()
        y_train_accuracy = cross_val_score(model_object,X_train2,y_train2,scoring='accuracy',cv=10,n_jobs=-1).mean()
        end_time=time.perf_counter()
        print('GA time= {} cpu time units*********************************************************************************'.format(end_time-start))

        acc_train.append(y_train_accuracy)
        print('ga train accacc with model trained on {}% of training data='.format(str(100)),y_train_accuracy)
        y_test_pred = model_object.predict(X_test)
        # y_test_accuracy=cross_val_score(model_object,X_test,scoring='accuracy',cv=10,n_jobs=-1).mean()
        y_test_accuracy = accuracy_score(y_test, y_test_pred)
        acc_test.append(y_test_accuracy)
        print('ga test acc with model trained on {}% of training data='.format(str(100)),y_test_accuracy)
        loss_ga=pd.DataFrame({'Loss_value':model_object.fitness_curve})
        loss_ga.to_csv('NN/ga'+'_loss_curve.csv')
        lc_ga=pd.DataFrame({'Train Acuuracy':acc_train,'Test Accuracy':acc_test,'% of training data':[20,40,60,80,100]})
        lc_ga.to_csv('NN/ga'+'_learning_curve.csv')
        # print(model_object.fitness_curve)
        # print(model_object.curve)  
        print('*******************************************************************************************************************')

    def evaluate_backprop(self,X, y, X_train, X_test, y_train, y_test):
        acc_train=[]
        acc_test=[]
        backprop_model = mlrose.NeuralNetwork(random_state=1, 
                                        algorithm ='gradient_descent', 
                                        bias = False, is_classifier = True, 
                                         early_stopping = True, 
                                        clip_max = 5, max_attempts = 5000,curve=True,max_iters=5000,\
                                            learning_rate=0.001,hidden_nodes=(5,5),\
                                                activation='relu')
        percent=0.80
        while percent >=0.20:
            X_train2, X_test2, y_train2, y_test2 = train_test_split(
                X_train, y_train, random_state=0, test_size=percent)
            model_object=backprop_model.fit(X_train2,y_train2)
            # score=cross_validate(model_object,X,y,scoring='accuracy',cv=10,n_jobs=-1,return_train_score=True)

            # y_train_pred=backprop_model.predict(X_train2)
            y_train_accuracy = cross_val_score(model_object,X_train2,y_train2,scoring='accuracy',cv=10,n_jobs=-1).mean()

            # y_train_accuracy = accuracy_score(y_train2, y_train_pred)
            acc_train.append(y_train_accuracy)
            print('backprop train acc with model trained on {}% of training data='.format(str(math.ceil((1-percent)*100))),y_train_accuracy)
            y_test_pred = model_object.predict(X_test)
            # y_test_accuracy=cross_val_score(model_object,X_test,scoring='accuracy',cv=10,n_jobs=-1).mean()
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            acc_test.append(y_test_accuracy)
            print('backprop test acc with model trained on {}% of training data='.format(str(math.ceil((1-percent)*100))),y_test_accuracy)
            # loss_backprop=pd.DataFrame({'Loss_value':model_object.fitness_curve})
            # loss_backprop.to_csv('NN/backprop'+'_loss_curve.csv')
            # print("backprop curve"model_object.fitness_curve)

            percent-=0.2 
        model_object=backprop_model.fit(X_train,y_train)
        # y_train_pred=backprop_model.predict(X_train)
        start=time.perf_counter()
        y_train_accuracy = cross_val_score(model_object,X_train,y_train,scoring='accuracy',cv=10,n_jobs=-1).mean()
        end_time=time.perf_counter()
        print('Backprop time= {} cpu time units*********************************************************************************'.format(end_time-start))
        # y_train_accuracy = accuracy_score(y_train, y_train_pred)
        acc_train.append(y_train_accuracy)
        print('backprop train accacc with model trained on {}% of training data='.format(str(100)),y_train_accuracy)
        y_test_pred = model_object.predict(X_test)
        # y_test_accuracy=cross_val_score(model_object,X_test,scoring='accuracy',cv=10,n_jobs=-1).mean()
        y_test_accuracy = accuracy_score(y_test, y_test_pred)
        acc_test.append(y_test_accuracy)
        print('backprop test acc with model trained on {}% of training data='.format(str(100)),y_test_accuracy)
        loss_backprop=pd.DataFrame({'Loss_value':model_object.fitness_curve})
        loss_backprop.to_csv('NN/backprop'+'_loss_curve.csv')
        lc_backprop=pd.DataFrame({'Train Acuuracy':acc_train,'Test Accuracy':acc_test,'% of training data':[20,40,60,80,100]})
        lc_backprop.to_csv('NN/backprop'+'_learning_curve.csv')
        # print(model_object.fitness_curve)
        # print(model_object.curve)        
        print('*******************************************************************************************************************')

    def evaluate_optimized(self,alg):
        X, y, X_train, X_test, y_train, y_test = self.get_data()
        if alg=='back_prop':
            self.evaluate_backprop(X, y, X_train, X_test, y_train, y_test)
        elif alg=='SA':
            self.evaluate_sa(X, y, X_train, X_test, y_train, y_test)
        elif alg=='GA':
            self.evaluate_ga(X, y, X_train, X_test, y_train, y_test)
        elif alg=='RHC':
            self.evaluate_rhc(X, y, X_train, X_test, y_train, y_test)

    def evaluate_NN_alg(self):
        # alg=['SA','back_prop','GA','RHC']
        alg=['back_prop']
        with concurrent.futures.ProcessPoolExecutor() as f:
            f.map(self.evaluate_optimized,alg)

    def get_data(self):
        X, y = load_breast_cancer(return_X_y=True)
        # print(X)
        # Encoding categorical data values
        labelencoder_Y = LabelEncoder()
        y = labelencoder_Y.fit_transform(y)
        y = np.absolute(y-1)
        # print(z)
        robust_scaler = RobustScaler()
        X = robust_scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=0, test_size=0.20)
        # sc = StandardScaler()
        # X_train = sc.fit_transform(X_train)
        # X_test = sc.transform(X_test)
        # Details of data
        test_size = X_test.shape[0]
        train_size = X_train.shape[0]
        print('Train data size: ', train_size)
        print('Test data size: ', test_size)
        print('% of train test split :', str(80)+'-'+str(20))
        print('1 in test: ', len(y_test[y_test == 1]))
        print('1 in train: ', len(y_train[y_train == 1]))

        print(X.shape)
        print(y.shape)
        print(X_train.shape)
        print(y_train.shape)
        return X, y, X_train, X_test, y_train, y_test




if __name__ == "__main__":
    prob_size=5
    obj=NN(prob_size)
    # obj.optimize()
    # obj.evaluate_optimized()
    obj.evaluate_NN_alg()

   