import mlrose_hiive as mlrose
import numpy as np
import concurrent.futures
import pandas as pd
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

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 
np.random.seed(3341)
problem_size = 25


class optimization_problem:
    def __init__(self, name):
        print(name + ' optimization')
        self.noOfiteration=10000

    def gridSearchSA(self,problem,problem_name,problem_size_space,iteration):
        sam=sa.SARunner(problem=problem,experiment_name='SA_Annealing',seed=1,iteration_list=[x for x in range(0,iteration+1,1000)],\
            max_attempts=5000 ,temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
        df_run_stats, df_run_curves = sam.run() 
        size=str(problem_size_space)
        df_run_curves.to_csv(size+'/sa_curves_'+problem_name+'_'+size+'.csv')
        df_run_stats.to_csv(size+'/sa_stats_'+problem_name+'_'+size+'.csv')
        print("********************************************************************************************************************************************************************************")

    def gridSearchRHC(self,problem,problem_name,problem_size_space,iteration):
        step=problem_size_space//2
        sam=rhc.RHCRunner(problem=problem,experiment_name='RHC',seed=1,iteration_list=[x for x in range(0,iteration+1,1000)],\
            max_attempts=5000 ,restart_list=[x for x in range(0,problem_size_space+1,step)] )
        df_run_stats, df_run_curves = sam.run() 
        size=str(problem_size_space)
        df_run_curves.to_csv(size+'/RHC_curves_'+problem_name+'_'+size+'.csv')
        df_run_stats.to_csv(size+'/RHC_stats_'+problem_name+'_'+size+'.csv')
        print("********************************************************************************************************************************************************************************")

    def gridSearchGA(self,problem,problem_name,problem_size_space,iteration):
        problem_size_space_exp=problem_size_space*10
        step=problem_size_space_exp//2
        sam=ga.GARunner(problem=problem,experiment_name='GA',seed=1,iteration_list=[x for x in range(0,iteration+1,1000)],\
            max_attempts=1000 ,population_sizes=[x for x in range(step,problem_size_space_exp+1,step)],mutation_rates=np.arange(.1,.7,.2) )  
        df_run_stats, df_run_curves = sam.run() 
        size=str(problem_size_space)
        df_run_curves.to_csv(size+'/GA_curves_'+problem_name+'_'+size+'.csv')
        df_run_stats.to_csv(size+'/GA_stats_'+problem_name+'_'+size+'.csv')
        print("********************************************************************************************************************************************************************************")

    def gridSearchMIMIC(self,problem,problem_name,problem_size_space,iteration):
        problem_size_space_exp=problem_size_space*20*4
        step=problem_size_space_exp//2
        sam=mimic.MIMICRunner(problem=problem,experiment_name='MIMIC',seed=1,iteration_list=[x for x in range(0,iteration+1,1000)],\
            # max_attempts=500 ,population_sizes=[x for x in range(step,problem_size_space_exp+1,step)],use_fast_mimic=True,keep_percent_list=[.1,0.5,0.75] )  
            max_attempts=500 ,population_sizes=[1600,3200],use_fast_mimic=True,keep_percent_list=[.1,0.5,0.75] )
        df_run_stats, df_run_curves = sam.run() 
        size=str(problem_size_space)
        df_run_curves.to_csv(size+'/MIMIC_curves_'+problem_name+'_'+size+'.csv')
        df_run_stats.to_csv(size+'/MIMIC_stats_'+problem_name+'_'+size+'.csv')
        print("********************************************************************************************************************************************************************************")
        

#*******************************************************************************************TSP*****************************************************************************************************

class TSP(optimization_problem):
    def __init__(self, size):
        super().__init__('TSP')
        self.problem_size = size
        
    def optimize(self):
        problem_size_space = self.problem_size
        # Initializing the problem
        init_state=[]
        length = problem_size_space//5
        for row in range(5):
            for col in range(length):
                init_state.append((row,col))
        problem = mlrose.TSPOpt(
            length=problem_size_space, maximize=False, coords=init_state)
        # SA
        # super().gridSearchSA(problem,'TSP',problem_size_space,self.noOfiteration)
        # RHC
        # super().gridSearchRHC(problem,'TSP',problem_size_space,self.noOfiteration)
        #GA
        # super().gridSearchGA(problem,'TSP',problem_size_space,self.noOfiteration)
        #MIMIC
        super().gridSearchMIMIC(problem,'TSP',problem_size_space,self.noOfiteration)

#*******************************************************************************************NQUEENS*****************************************************************************************************

class NQueens(optimization_problem):
    def __init__(self, size):
        super().__init__('NQueen')
        self.problem_size = size

    def optimize(self):
        problem_size_space = self.problem_size

        # Initializing the problem
        init_state = [ i for i in range(problem_size_space)]
        fitness = mlrose.Queens()
        problem = mlrose.DiscreteOpt(
            length=problem_size_space, fitness_fn=fitness, maximize=False, max_val=problem_size_space)
        # SA
        # super().gridSearchSA(problem,'NQueens',problem_size_space,self.noOfiteration)
        # RHC
        # super().gridSearchRHC(problem,'NQueens',problem_size_space,self.noOfiteration)
        #GA
        # super().gridSearchGA(problem,'NQueens',problem_size_space,self.noOfiteration)
        #MIMIC
        super().gridSearchMIMIC(problem,'NQueens',problem_size_space,self.noOfiteration)
 
#*******************************************************************************************4PEAKS*****************************************************************************************************

class Four_Peaks(optimization_problem):
    def __init__(self, size):
        super().__init__('4PeaksPeaks')
        self.problem_size = size

    def optimize(self):
        problem_size_space = self.problem_size
        # Initializing the problem
        init_state = np.random.randint(2,size=problem_size_space)
        fitness = mlrose.FourPeaks(t_pct=0.1)
        problem = mlrose.DiscreteOpt(
            length=problem_size_space, fitness_fn=fitness, maximize=True, max_val=2)
        # SA
        # super().gridSearchSA(problem,'4Peaks',problem_size_space,self.noOfiteration)
        # RHC
        # super().gridSearchRHC(problem,'4Peaks',problem_size_space,self.noOfiteration)
        #GA
        # super().gridSearchGA(problem,'4Peaks',problem_size_space,self.noOfiteration)
        #MIMIC
        super().gridSearchMIMIC(problem,'4Peaks',problem_size_space,self.noOfiteration)

#*******************************************************************************************KNAPSACK*****************************************************************************************************
class Knapsack(optimization_problem):
    def __init__(self, size):
        super().__init__('Knapsack')
        self.problem_size = size
        
    def optimize(self):
        problem_size_space = self.problem_size
        # Initializing the problem
        init_state = np.random.randint(0, 3, size=problem_size_space)
        weights = [int(np.random.randint(1, problem_size_space/2)) for _ in range(problem_size_space)]
        values = [int(np.random.randint(1, problem_size_space/2)) for _ in range(problem_size_space)]
        # print('weight:',weights)
        # print('value:',values)

        fitness = mlrose.Knapsack(weights=weights,values=values,max_weight_pct=1.0)#max_weight_pct (float, default: 0.35) – Parameter used to set maximum capacity of knapsack (W) as a percentage of the total of the weights list (W = max_weight_pct \times total_weight).
        problem = mlrose.DiscreteOpt(length=problem_size_space, fitness_fn=fitness, maximize=True, max_val=2)
        # SA
        # super().gridSearchSA(problem,'Knapsack',problem_size_space,self.noOfiteration)
        # RHC
        # super().gridSearchRHC(problem,'Knapsack',problem_size_space,self.noOfiteration)
        #GA
        # super().gridSearchGA(problem,'Knapsack',problem_size_space,self.noOfiteration)
        #MIMIC
        super().gridSearchMIMIC(problem,'Knapsack',problem_size_space,self.noOfiteration)


def executeParallel(obj):
    obj.optimize()
if __name__ == "__main__":
    prob_size=40
    # start=time.perf_counter()
    # for i in range(5,10,10):
    #     print('size===========',i)
    opt = TSP(40)
    opt.optimize()

    #     opt3 = Four_Peaks(i)
    #     opt3.optimize()
    #     opt4 = Knapsack(i)
    #     opt4.optimize()
    #     opt2 = NQueens(i)
    #     opt2.optimize()
    # finish=time.perf_counter()
    # print (f'The series execution took {(finish-start)} seconds')
#Parallel

    # start=time.perf_counter()
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     p_size=[TSP(prob_size),NQueens(prob_size),Knapsack(prob_size),Four_Peaks(prob_size)]
    #     results=executor.map(executeParallel,p_size)
    # finish=time.perf_counter()
    # print (f'The parallel execution took {(finish-start)} seconds')