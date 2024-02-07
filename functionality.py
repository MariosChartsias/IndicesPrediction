#####################################################functionality.py###########################################################

#####################################################
# Import Libraries
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
import math
import time
from datetime import datetime
import numpy as np
from numpy.random import choice
from sklearn import linear_model
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pygad
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import logging
from IPython.display import clear_output

#####################################################


#####################################################
# Set Options
#####################################################

# np.set_printoptions(linewidth=np.inf)  # print with no wrap the values
# np.set_printoptions(threshold=np.inf)  # print the whole results without the dots ...

pd.set_option('display.max_columns', None)  # Display all columns
# pd.set_option('display.max_rows', None)     # Display all rows

# 3.00000000e+01 -> 30
np.set_printoptions(precision=2)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

#####################################################
# Utility Function
#####################################################

def print_red_bold(text):
    print("\033[91;1m" + text + "\033[0m")
    
def print_modify_string_functions(expression):
    new_expression= expression.replace("+ -", "-")
    #print(f'before {new_expression}')
    new_expression2 = re.sub(r"\+ 0\.00\*\w+\^*[1|2]*|\-0\.00\*\w+\^*[1|2]*", "", new_expression)
    new_expression3=new_expression2 = re.sub(r"\s{2,}", "", new_expression2)
    print(new_expression3,"\n")    
    
    
##################################################### 
#Definitions, Initializations & Functions

O_array = np.array([1,-1,1,-1,1,-1,1,1,1,1]) #(10,1)
C_array = np.array([1,-1,1,-1,1,-1,1,-1,1,1])  #(10,1)
E_array = np.array([1,-1,1,-1,1,-1,1,-1,1,-1])  #(10,1)
A_array = np.array([-1,1,-1,1,-1,1,-1,1,1,1])  #(10,1)
N_array = np.array([-1,1,-1,1,-1,-1,-1,-1,-1,-1])  #(10,1)


Oinit=8
Cinit=14
Einit=20
Ainit=14
Ninit=38

answer_columns = ['OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5', 'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10',
                  'CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7', 'CSN8', 'CSN9', 'CSN10',
                  'EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9', 'EXT10',
                  'AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7', 'AGR8', 'AGR9', 'AGR10',    
                  'NRT1', 'NRT2', 'NRT3', 'NRT4', 'NRT5', 'NRT6', 'NRT7', 'NRT8', 'NRT9', 'NRT10']

answer_columns_without_OPN=['CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7', 'CSN8', 'CSN9', 'CSN10',
                            'EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9', 'EXT10',
                            'AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7', 'AGR8', 'AGR9', 'AGR10',    
                            'NRT1', 'NRT2', 'NRT3', 'NRT4', 'NRT5', 'NRT6', 'NRT7', 'NRT8', 'NRT9', 'NRT10']

answer_columns_without_CSN=['OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5', 'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10',
                            'EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9', 'EXT10',
                            'AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7', 'AGR8', 'AGR9', 'AGR10',    
                            'NRT1', 'NRT2', 'NRT3', 'NRT4', 'NRT5', 'NRT6', 'NRT7', 'NRT8', 'NRT9', 'NRT10']

answer_columns_without_EXT=['OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5', 'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10',
                            'CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7', 'CSN8', 'CSN9', 'CSN10',
                            'AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7', 'AGR8', 'AGR9', 'AGR10',    
                            'NRT1', 'NRT2', 'NRT3', 'NRT4', 'NRT5', 'NRT6', 'NRT7', 'NRT8', 'NRT9', 'NRT10']

answer_columns_without_AGR=['OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5', 'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10',
                            'CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7', 'CSN8', 'CSN9', 'CSN10',
                            'EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9', 'EXT10',
                            'NRT1', 'NRT2', 'NRT3', 'NRT4', 'NRT5', 'NRT6', 'NRT7', 'NRT8', 'NRT9', 'NRT10']

answer_columns_without_NRT=  ['OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5', 'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10',
                              'CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7', 'CSN8', 'CSN9', 'CSN10',
                              'EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9', 'EXT10',
                              'AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7', 'AGR8', 'AGR9', 'AGR10']

column_names = ['OPN', 'CSN', 'EXT', 'AGR', 'NRT']
column_names_without_OPN = ['CSN', 'EXT', 'AGR', 'NRT']
column_names_without_CSN = ['OPN', 'EXT', 'AGR', 'NRT']
column_names_without_EXT = ['OPN', 'CSN', 'AGR', 'NRT']
column_names_without_AGR = ['OPN', 'CSN', 'EXT', 'NRT']
column_names_without_NRT = ['OPN', 'CSN', 'EXT', 'AGR']
score_columns = ['O','C','E','A','N']

 
# Define a dictionary to store your MSE values
mse = {
    'mseOO': 'mse:O↦O', #mean squared error of index O with answers OPN1..OPN10
    'mseCC': 'mse:C↦C', #mean squared error of index C with answers CSN1..CSN10
    'mseEE': 'mse:E↦E', #mean squared error of index E with answers EXT1..EXT10
    'mseAA': 'mse:A↦A', #mean squared error of index A with answers AGR1..AGR10
    'mseNN': 'mse:N↦N', #mean squared error of index N with answers NRT1..NRT10
    'mseCO': 'mse:C↦O', #mean squared error of index O with answers CSN1..CSN10
    'mseEO': 'mse:E↦O', #mean squared error of index O with answers EXT1..EXT10
    'mseAO': 'mse:A↦O', #mean squared error of index O with answers AGR1..AGR10
    'mseNO': 'mse:N↦O', #mean squared error of index O with answers NRT1..NRT10
    'mseOC': 'mse:O↦C', #mean squared error of index C with answers CSN1..CSN10
    'mseEC': 'mse:E↦C', #mean squared error of index C with answers EXT1..EXT10
    'mseAC': 'mse:A↦C', #mean squared error of index C with answers AGR1..AGR10
    'mseNC': 'mse:N↦C', #mean squared error of index C with answers NRT1..NRT10
    'mseOE': 'mse:O↦E', #mean squared error of index E with answers OPN1..OPN10
    'mseCE': 'mse:C↦E', #mean squared error of index E with answers CSN1..CSN10
    'mseAE': 'mse:A↦E', #mean squared error of index E with answers AGR1..AGR10
    'mseNE': 'mse:N↦E', #mean squared error of index E with answers NRT1..NRT10
    'mseOA': 'mse:O↦A', #mean squared error of index A with answers OPN1..OPN10
    'mseCA': 'mse:C↦A', #mean squared error of index A with answers CSN1..CSN10
    'mseEA': 'mse:E↦A', #mean squared error of index A with answers EXT1..EXT10
    'mseNA': 'mse:N↦A', #mean squared error of index A with answers NRT1..NRT10
    'mseON': 'mse:O↦N', #mean squared error of index N with answers OPN1..OPN10
    'mseCN': 'mse:C↦N', #mean squared error of index N with answers CSN1..CSN10
    'mseEN': 'mse:E↦N', #mean squared error of index N with answers EXT1..EXT10
    'mseAN': 'mse:A↦N', #mean squared error of index N with answers AGR1..AGR10

}

# Access the MSE values from the dictionary
mseOO = mse['mseOO']
mseCC = mse['mseCC']
mseEE = mse['mseEE']
mseAA = mse['mseAA']
mseNN = mse['mseNN']
mseCO = mse['mseCO']
mseEO = mse['mseEO']
mseAO = mse['mseAO']
mseNO = mse['mseNO']
mseOC = mse['mseOC']
mseEC = mse['mseEC']
mseAC = mse['mseAC']
mseNC = mse['mseNC']
mseOE = mse['mseOE']
mseCE = mse['mseCE']
mseAE = mse['mseAE']
mseNE = mse['mseNE']
mseOA = mse['mseOA']
mseCA = mse['mseCA']
mseEA = mse['mseEA']
mseNA = mse['mseNA']
mseON = mse['mseON']
mseCN = mse['mseCN']
mseEN = mse['mseEN']
mseAN = mse['mseAN']

mse_v2 = {
    'mseO': 'mse:CEAN↦O', #mean squared error of index O without answers OPN1..OPN10
    'mseC': 'mse:OEAN↦C', #mean squared error of index O without answers CSN1..CSN10
    'mseE': 'mse:OCAN↦E', #mean squared error of index O without answers EXT1..EXT10
    'mseA': 'mse:OCEN↦A', #mean squared error of index O without answers AGR1..AGR10
    'mseN': 'mse:OCEA↦N', #mean squared error of index O without answers NRT1..NRT10
}

mseO = mse_v2['mseO']
mseC = mse_v2['mseC']
mseE = mse_v2['mseE']
mseA = mse_v2['mseA']
mseN = mse_v2['mseN']


def count_lines(array):
    if isinstance(array, np.ndarray):
        return array.shape[0]  # Return the number of rows
    else:
        return 0  # Return 0 if the input is not a valid numpy array
def count_columns(array):
    if isinstance(array, np.ndarray):
        return array.shape[1]  # Return the number of columns
    else:
        return 0  # Return 0 if the input is not a valid numpy array

def combine_arrays(arr1, arr2, arr3, arr4, arr5):
    if all(isinstance(arr, np.ndarray) for arr in [arr1, arr2, arr3, arr4, arr5]):
        if all(arr.shape[0] == arr1.shape[0] for arr in [arr2, arr3, arr4, arr5]):
            combined_array = np.concatenate((arr1, arr2, arr3, arr4, arr5), axis=1)
            return combined_array
        else:
            return None  # Return None if the arrays have different number of rows
    else:
        return None  # Return None if the inputs are not valid numpy arrays

def stop_time(start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nExecution time: {elapsed_time:3.2f} seconds\n")    
    
    
def print_metrics(y_true, y_pred):
    # Calculate the accuracy of the model (R-squared score)
    accuracy = r2_score(y_true, y_pred) * 100

    # Calculate mean absolute error (MAE)
    mae = mean_absolute_error(y_true, y_pred)

    # Calculate mean squared error (MSE)
    mse = mean_squared_error(y_true, y_pred)**2
    print(f"Accuracy of the model: %.2f%%, Mean Absolute Error: %.2f, Mean Squared Error: %.2f" % (accuracy, mae, mse))
    
def calculate_metrics(y_true, y_pred):
    # Calculate the accuracy of the model (R-squared score)
    accuracy = r2_score(y_true, y_pred) * 100

    # Calculate mean absolute error (MAE)
    mae = mean_absolute_error(y_true, y_pred)

    # Calculate mean squared error (MSE)
    mse = mean_squared_error(y_true, y_pred)**2
    
    return accuracy,mae,mse

global progress_of_optimization_without_answers
progress_of_optimization_without_answers = {"O" : [],"C": [],"E": [],"A": [],"N": []} 


#####################################################

#####################################################

## def: Calculate index: 

#- Openness(O)
#- Conscientiousness(C)
#- Extroversion(E)
#- Agreeableness(A)
#- Neuroticism(N)

def Result_O(array, O_array, progress_bar):
    if isinstance(array, np.ndarray) and isinstance(O_array, np.ndarray):
        n = array.shape[0]  # Get the number of rows in the array
        # Use np.dot for matrix multiplication and then add Oinit
        output = np.dot(array[:, :10], O_array) + Oinit
        output = output.reshape(-1, 1)  # Reshape to (n, 1)
        progress_bar.update(1)
        return output
    else:
        return None  # Return None if the inputs are not valid numpy arrays
def Result_C(array, C_array, progress_bar):
    if isinstance(array, np.ndarray) and isinstance(C_array, np.ndarray):
        n = array.shape[0]  # Get the number of rows in the array
        # Use np.dot for matrix multiplication and then add Oinit
        output = np.dot(array[:, 10:20], C_array) + Cinit
        output = output.reshape(-1, 1)  # Reshape to (n, 1)
        progress_bar.update(1)
        return output
    else:
        return None  # Return None if the inputs are not valid numpy arrays
def Result_E(array, E_array, progress_bar):
    if isinstance(array, np.ndarray) and isinstance(E_array, np.ndarray):
        n = array.shape[0]  # Get the number of rows in the array
        # Use np.dot for matrix multiplication and then add Oinit
        output = np.dot(array[:, 20:30], E_array) + Einit
        output = output.reshape(-1, 1)  # Reshape to (n, 1)
        progress_bar.update(1)
        return output
    else:
        return None  # Return None if the inputs are not valid numpy arrays
def Result_A(array, A_array, progress_bar):
    if isinstance(array, np.ndarray) and isinstance(A_array, np.ndarray):
        n = array.shape[0]  # Get the number of rows in the array
        # Use np.dot for matrix multiplication and then add Oinit
        output = np.dot(array[:, 30:40], A_array) + Ainit
        output = output.reshape(-1, 1)  # Reshape to (n, 1)
        progress_bar.update(1)
        return output
    else:
        return None  # Return None if the inputs are not valid numpy arrays
def Result_N(array, N_array, progress_bar):
    if isinstance(array, np.ndarray) and isinstance(N_array, np.ndarray):
        n = array.shape[0]  # Get the number of rows in the array
        # Use np.dot for matrix multiplication and then add Oinit
        output = np.dot(array[:, 40:50], N_array) + Ninit
        output = output.reshape(-1, 1)  # Reshape to (n, 1)
        progress_bar.update(1)
        return output
    else:
        return None  # Return None if the inputs are not valid numpy arrays
    
    
#####################################################    


#####################################################
#GENETIC ALGORYTHM LINEAR FUNCTIONS
#####################################################

answers = np.array([])
OCEAN = []

def getOCEAN(score_O, score_C, score_E, score_A, score_N):
    global OCEAN
    OCEAN = combine_arrays(score_O, score_C, score_E, score_A, score_N)
    return np.array(OCEAN)

OCEAN = np.array(OCEAN)

    
def getResults(nrows=1000):
    results = pd.read_csv('DataFiles/Q&Adata.csv', nrows=40000)
    answers = results[answer_columns].to_numpy()
    return answers
    

new_dimension=500 #specifies the size of population

class Genetic_algorythm_Linear_Fitness:
    def __init__(self, OCEAN, answers):
        self.OCEAN = OCEAN
        self.answers=answers

    # Define events for synchronization
    crossover_event = threading.Event()
    mutation_event = threading.Event()

    
    def reset_events(self):
        crossover_event.clear()
        mutation_event.clear()


    logging.basicConfig(level=logging.DEBUG)

    def getSpecificPopulation(self,x=answers.shape[0], y=40, num_ones_per_row=20):
        # Generate an array with zeros
        population = np.zeros((x, y), dtype=int)

        # Set exactly num_ones_per_row ones in each row
        for i in range(x):
            population[i, np.random.choice(y, size=num_ones_per_row, replace=False)] = 1

        return population

    def getPopulation(self,x=answers.shape[0],y=40):
        return np.random.choice([1, 0], size=(x, 40))


        # Generate random indices for ones in each row
        row_indices = np.arange(x)
        #print(row_indices)
        col_indices = np.random.choice(y, size=(x, num_ones_per_row), replace=False)
        #print(col_indices)

        # Set ones in the specified positions
        population[row_indices[:, np.newaxis], col_indices] = 1

        return population

    def loadPopulation(self, index,x=answers.shape[0],y=40):
        #print(f"loadPopulation -> x = answers.shape[0] = {x}, self.answers.shape[0]={self.answers.shape[0]}")
        index_paths = {
            "O": ('DataFiles/population_without_O.csv'),
            "C": ('DataFiles/population_without_C.csv'),
            "E": ('DataFiles/population_without_E.csv'),
            "A": ('DataFiles/population_without_A.csv'),
            "N": ('DataFiles/population_without_N.csv'),
        }
        if os.path.exists(index_paths[index]):
            existing_array = np.loadtxt(index_paths[index], delimiter=',', skiprows=1)
            #print(f" existing_array.shape= {existing_array.shape} , x= {x}")
            if existing_array.shape[0]==x:
                return existing_array
            elif existing_array.shape[0]<x:
                #print("condition=existing_array.shape[0]<x 1")
                x=x-existing_array.shape[0]
                final_array = np.concatenate((existing_array, np.random.choice([1, 0], size=(x, 40))), axis=0)
                print(f"final_array = {final_array.shape} , population.shape={population.shape}")
                return final_array
            elif existing_array.shape[0]>x:
                #print("condition=existing_array.shape[0]>x 2 ")
                final_array = np.random.choice([1, 0], size=(x, 40))
                #final_array= np.ones((x, 40)) # only with 1
                #print(f"final_array = {final_array.shape} , population.shape={population.shape}")
                return final_array
        else:
            return np.random.choice([1, 0], size=(x, 40))

    def savePopulation(self,index,population):
        index_paths = {
        "O": ('DataFiles/population_without_O.csv',answer_columns_without_OPN),
        "C": ('DataFiles/population_without_C.csv',answer_columns_without_CSN),
        "E": ('DataFiles/population_without_E.csv',answer_columns_without_EXT),
        "A": ('DataFiles/population_without_A.csv',answer_columns_without_AGR),
        "N": ('DataFiles/population_without_N.csv',answer_columns_without_NRT),
        }
        path=index_paths[index][0]
        header=index_paths[index][1]
        np.savetxt(path, population, delimiter=',', header=','.join(header), comments='')

    def loadFitness(self,index, population):
        index_paths = {
                "O": ('DataFiles/fitness_array_O.csv'),
                "C": ('DataFiles/fitness_array_C.csv'),
                "E": ('DataFiles/fitness_array_E.csv'),
                "A": ('DataFiles/fitness_array_A.csv'),
                "N": ('DataFiles/fitness_array_N.csv'),
            }
        if os.path.exists(index_paths[index]):
            existing_array = np.loadtxt(index_paths[index], delimiter=',', skiprows=1).reshape(-1,1)
            if existing_array.shape[0]==population.shape[0]:
                return existing_array
            elif population.shape[0]==1:
                final_array = self.Fitness_vector(index,population)
                #print(f"final_array = Fitness_vector(index,population).shape = {final_array.shape}, population.shape= {population.shape}")
                return final_array
            elif population.shape[0]>1:
                final_array = self.Fitness(index,population)
                #print(f"final_array = Fitness(index,population).shape = {final_array.shape}, population.shape= {population.shape}")
                return final_array
        else:
            return self.Fitness(index,population)

    def saveFitness(self,index,fitness_array):
        index_saveFitness = {
                "O": ('DataFiles/fitness_array_O.csv','mse_O'),
                "C": ('DataFiles/fitness_array_C.csv','mse_C'),
                "E": ('DataFiles/fitness_array_E.csv','mse_E'),
                "A": ('DataFiles/fitness_array_A.csv','mse_A'),
                "N": ('DataFiles/fitness_array_N.csv','mse_N'),
        }
        path=index_saveFitness[index][0]
        header=index_saveFitness[index][1]
        np.savetxt(path, fitness_array, delimiter=',', header=header)



    def Fitness_vector(self,index,array1,array2=None):
        array2 = array2 if array2 is not None else self.OCEAN
        #print(f'Fitness_vector =  array2.shape{array2.shape}, OCEAN.shape={OCEAN.shape}')
        array1=array1.reshape(1,-1)
        size_x=OCEAN.shape[0]
        x=array1
        #print(f"Fitness_vector / array1.shape = {array1.shape} ")
        if(array1.shape[1]==40):
            fitness_array = np.empty((size_x, 1))
            x=np.tile(array1, (size_x, 1))
            if(index=="O"):
                y=array2[:size_x,:1]
                x=np.sum(x*self.answers[:size_x,10:],axis=1).reshape(-1,1)
                #print(f"Fitness_vector/if(index=='O'): y.shape = {y.shape} , x.shape={x.shape} ")
            elif(index=="C"):
                y=array2[:size_x,1:2]
                x=np.sum(x*np.delete(self.answers[:size_x,:], np.arange(10, 20), axis=1),axis=1).reshape(-1,1)
            elif(index=="E"):
                y=array2[:size_x,2:3]
                x=np.sum(x*np.delete(self.answers[:size_x,:], np.arange(20, 30), axis=1),axis=1).reshape(-1,1)
            elif(index=="A"):
                y=array2[:size_x,3:4]
                x=np.sum(x*np.delete(self.answers[:size_x,:], np.arange(30, 40), axis=1),axis=1).reshape(-1,1)
            elif(index=="N"):
                y=array2[:size_x,4:]
                x=np.sum(x*np.delete(self.answers[:size_x,:], np.arange(40, 50), axis=1),axis=1).reshape(-1,1)
            else:
                return None
            fitness_vector = self.fitness(x,y)
            #print(f"Fitness_vector / fitness_vector.shape = {fitness_vector.shape} ")
            return fitness_vector

    def perform_fitness(self,x,i,size_x,size_x_population,fitness_array,array1,array2,index):
        x=np.tile(array1[i,:], (size_x, 1))
        if(index=="O"):
            y=array2[:size_x,:1]
            x=np.sum(x*self.answers[:size_x,10:],axis=1).reshape(-1,1)
        elif(index=="C"):
            y=array2[:size_x,1:2]
            x=np.sum(x*np.delete(self.answers[:size_x,:], np.arange(10, 20), axis=1),axis=1).reshape(-1,1)
        elif(index=="E"):
            y=array2[:size_x,2:3]
            x=np.sum(x*np.delete(self.answers[:size_x,:], np.arange(20, 30), axis=1),axis=1).reshape(-1,1)
        elif(index=="A"):
            y=array2[:size_x,3:4]
            x=np.sum(x*np.delete(self.answers[:size_x,:], np.arange(30, 40), axis=1),axis=1).reshape(-1,1)
        elif(index=="N"):
            y=array2[:size_x,4:]
            x=np.sum(x*np.delete(self.answers[:size_x,:], np.arange(40, 50), axis=1),axis=1).reshape(-1,1)
        else:
            return None
        fitness_array[i]=self.fitness(x,y)    

    def Fitness(self,index,array1,array2=None):
        array2 = array2 if array2 is not None else self.OCEAN
        #print(f'Fitness array2.shape{array2.shape}')
        size_x=OCEAN.shape[0]
        size_x_population=array1.shape[0]
        x=array1
        #print(f"array1.shape = {array1.shape}")
        if(array1.shape[0]==10):
            x=x.reshape(1,10) 
            x=np.tile(array1, (OCEAN.shape[0], 1)) 
            if(index=="O"):
                y=array2[:x.shape[0],:1]
                x=x*self.answers[:,:10]
            elif(index=="C"):
                y=array2[:x.shape[0],1:2]
                x=x*self.answers[:,10:20]
            elif(index=="E"):
                y=array2[:x.shape[0],2:3]
                x=x*self.answers[:,20:30]
            elif(index=="A"):
                y=array2[:x.shape[0],3:4]
                x=x*self.answers[:,30:40]
            elif(index=="N"):
                y=array2[:x.shape[0],4:]
                x=x*self.answers[:,40:]
            return self.fitness(x,y)
        elif(array1.shape[0]==1 and array1.shape[1]==40):
            fitness_array = np.empty((size_x, 1))
            threads = []
            for i in range(0,size_x_population):
                thread = threading.Thread(target=self.perform_fitness, args=(x,i,size_x,size_x_population,fitness_array,array1,array2,index))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
        elif(array1.shape[1]==40):
            fitness_array = np.empty((size_x_population, 1))
            threads = []
            for i in range(0,size_x_population):
                thread = threading.Thread(target=self.perform_fitness, args=(x,i,size_x,size_x_population,fitness_array,array1,array2,index))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

        return fitness_array

    def fitness(self,x,y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        Linreg = LinearRegression()
        Linreg.fit(x_train, y_train)
        # Make predictions on the test set
        y_pred = Linreg.predict(x_test)
        #print_metrics(y_test,y_pred)
        #print("x_shape:",x.shape)
        #print("y_shape:",y.shape)    
        #print("y_train:", y_train.shape)
        #print("x_train:", x_train.shape)
        #print("y_test:", y_test.shape)
        #print("x_test:", x_test.shape)
        mse = mean_squared_error(y_test, y_pred)
        #print(f"fitness(x,y).shape={mse.shape}")
        return mse

    def sort_arrays(self,population, fitness):
        sorted_indices = np.argsort(fitness, axis=0)
        #print(f"sorted_indices = {sorted_indices}, sorted_indices.shape = {sorted_indices.shape}")
        if(population.shape[0]>1):
            sorted_population = population[sorted_indices[:, 0]]
            sorted_fitness = np.squeeze(fitness[sorted_indices])
        else:
            sorted_population=population
            sorted_fitness = fitness


        return sorted_population, sorted_fitness.reshape(population.shape[0],1)


    def perform_crossover(self,index,population_copy, row, avg, avg2):
        vector=population_copy[row]
        min_value = float('inf')
        for i in range(1,40): #40 must be replaced with population_copy.shape[1]-1
            vector = np.concatenate((population[row, i:], population[row, :i]), axis=0)
            avg2 = self.Fitness_vector(index, population_copy[row])
            if avg2 < min_value and avg2<avg :
                min_value=avg2
                vector = population_copy[row]
        self.crossover_event.set() # Signal that crossover is complete
        if(min_value<avg):
            population_copy[row]=vector
            #logging.debug(f"perform_crossover.success(index, row, min_value, avg)=({index, row, min_value, avg})")
        else:
            pass
            #logging.debug(f"perform_crossover.fail(index, row, min_value, avg)=({index, row, min_value, avg})")


    def crossover(self,index,population, fitness_array):
        x, y = population.shape
        avg = np.mean(fitness_array)
        avg2 = avg
        rows = np.argwhere(fitness_array > avg)[:, 0]
        population_copy = np.copy(population)
        threads = []
        for row in rows:
            thread = threading.Thread(target=self.perform_crossover, args=(index,population_copy, row, avg, avg2))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        self.crossover_event.clear() # Clear the event for the next use   
        return population_copy


    def perform_mutation(self,index, population_copy, row, avg, fitness_array):
        # The original perform_mutation function remains unchanged
        vector = population_copy[row]
        min_value = float('inf')
        for y1 in range(40):
            y2 = np.random.randint(0, 39)
            while y1==y2:
                y2 = np.random.randint(0, 39)
            try:
                new_array = self.swap(population_copy.copy(), row, y1, y2)
                avg2 = self.Fitness_vector(index, new_array[row, :])
                #logging.debug(f"row={row},y1={y1},y2={y2},min_value={min_value}")
                if avg2 < min_value and avg2<avg:
                    min_value = avg2
                    vector = new_array[row]
            except Exception as e:
                logging.error(f"Exception in thread: {e},row={row}, y1={y1},y2={y2},min_value={min_value}")
        if min_value < avg:
            #logging.debug(f"perform_mutation.success(index, row, min_value, avg)=({index, row, min_value, avg})")
            population_copy[row] = vector
        else:
            #logging.debug(f"perform_mutation.fail(y1={y1},y2={y2},min_value={min_value}, index, row, min_value, avg)=({index, row, min_value, avg})")
            pass
        self.mutation_event.set()  # Signal that mutation is complete

    def swap(self,array, row, y1, y2):
        array[row, y1], array[row, y2] = array[row, y2], array[row, y1]
        return array

    def mutation(self,index, population, fitness_array):
        avg = np.mean(fitness_array)
        mut_rows_bad = np.argwhere(fitness_array > avg)[:, 0]
        population_copy = np.copy(population)
        threads = []
        counter = threading.Lock()
        total_rows = len(mut_rows_bad)

        for row in mut_rows_bad:
            thread = threading.Thread(target=self.perform_mutation,
                                      args=(index,
                                            population_copy,
                                            row,
                                            avg,
                                            fitness_array))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        self.mutation_event.clear()  # Clear the event for the next use
        return population_copy



    def genetic_algorythm(self,index,population,fitness_array,generations=1,top=0):
        if(top==0):
            for i in range(generations+1):
                print(f"Generation:{i}, vectors for optimization:",len(np.argwhere(fitness_array > np.mean(fitness_array))[:,0])," avg solution mse:",np.mean(fitness_array))
                print("crossover start")
                population=self.crossover(index,population,fitness_array)
                print("mutation start")
                population=self.mutation(index,population,fitness_array)
                fitness_array=Fitness(index,population)
                population,fitness_array = sort_arrays(population, fitness_array)
        elif(top==1):
            i=0
            avg=np.mean(fitness_array)
            vevtor_for_optimization=len(np.argwhere(fitness_array > avg)[:,0])
            vevtor_for_optimization_prev=population.shape[0]
            condition_of_optimization=vevtor_for_optimization>0
            crossover_loop=False
            mutation_loop=False
            condition_of_loop=crossover_loop and mutation_loop

            print(f"START vevtor_for_optimization prev={vevtor_for_optimization_prev},{vevtor_for_optimization},i={i}")
            population_prev=population.copy()
            population_prev2=population.copy()

            min = avg
            first=avg
            #prev = np.empty((size_x_population, 1))
            while condition_of_optimization and not condition_of_loop:
                start_time = datetime.now()
                print(f"Start Time: {start_time}")
                ###########METRICS############
                print("Generation:",i," vectors for optimization:",vevtor_for_optimization," avg solution mse:",avg)

                i=i+1

                if(not crossover_loop):
                    print("crossover start")
                    population=self.crossover(index,population,fitness_array)
                    crossover_loop=np.array_equal(population, population_prev)
                    population_prev2=self.Fitness(index,population)
                    print(f"crossover end with previous_avg > avg:{np.mean(population_prev2)>np.mean(self.Fitness(index,population_prev))}")
                    print(f"crossover_loop=np.array_equal(population, population_prev)={np.array_equal(population, population_prev)}")
                if(not mutation_loop):
                    print("mutation start")
                    population=self.mutation(index,population,fitness_array)
                    mutation_loop=np.array_equal(population, population_prev)
                    print(f"mutation end with previous_avg > avg:{np.mean(self.Fitness(index,population)) > np.mean(self.Fitness(index,population_prev))}")
                    print(f"mutation_loop=np.array_equal(population, population_prev)={np.array_equal(population, population_prev)}")

                condition_of_loop=crossover_loop and mutation_loop
                fitness_array=self.Fitness(index,population)
                population,fitness_array = self.sort_arrays(population, fitness_array)

                population_prev=population.copy()
                vevtor_for_optimization_prev=vevtor_for_optimization
                prevavg=avg
                avg=np.mean(fitness_array)
                vevtor_for_optimization=len(np.argwhere(fitness_array > avg)[:,0])
                condition_of_optimization=vevtor_for_optimization>0 #and vevtor_for_optimization_prev > vevtor_for_optimization

                print(f"condition_of_optimization={condition_of_optimization} and not condition_of_loop={not condition_of_loop}")
                print(f"before save: condition_of_save (avg<min) = {avg<min}, avg={avg},min={min}")
                ########################SAVE CONTDITION############################
                condition_of_save = avg<min
                if(condition_of_save):
                    self.savePopulation(index,population)
                    self.saveFitness(index,fitness_array)
                    min=avg
                    print("save: ",condition_of_save)
                    print("success:",((prevavg-avg)/(prevavg))*100,"%")
                ########################END SAVE CONTDITION############################




                #population_prev=population.copy()
                #vevtor_for_optimization_prev=vevtor_for_optimization
                #prevavg=avg
                #avg=np.mean(fitness_array)
                #vevtor_for_optimization=len(np.argwhere(fitness_array > avg)[:,0])
                #condition_of_optimization=vevtor_for_optimization>0 #and vevtor_for_optimization_prev > vevtor_for_optimization

                print(f"END vevtor_for_optimization prev={vevtor_for_optimization_prev},new={vevtor_for_optimization},i={i}")


                #########METRICS##############
                #print("condition: ", not np.array_equal(prev, fitness_array))
                end_time = datetime.now()
                # Calculate the time difference
                time_difference = end_time - start_time
                # Print the result
                print(f"End Time: {end_time}")
                print(f"Time Difference: {time_difference}")
                if(i>1000):break
                if(i==10 and first==avg):break

        return population,fitness_array


    def getFunction(self,x,y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        Linreg = LinearRegression()
        Linreg.fit(x_train, y_train)
        y_pred = Linreg.predict(x_test)

        function_coefficients = Linreg.coef_
        function_intercept = Linreg.intercept_

        # Print the function used by the model
        global arrayext
        arrayext=[]
        for i, target_variable in enumerate("O"):
            coefficients = function_coefficients[i]
            arrayext.append(function_coefficients[i])
            terms = [f"{coeff:.2f}*{answer_columns[j]}" for j, coeff in enumerate(coefficients)]
            terms_str = " + ".join(terms)
            function_str = f"Function for O = {function_intercept[i]:.2f} + {terms_str}"
            print(function_str)

    def start_process_of_genetic_algorythm(self):
        global population,fitness_array
        for i in score_columns:
            print_red_bold(f"*********************  genetic algorythm for index {i}  *********************************")
            population=self.loadPopulation(i,new_dimension,40)
            fitness_array=self.loadFitness(i,population)
            #print(f" fitness_array = {fitness_array.shape}")
            population,fitness_array = self.sort_arrays(population, fitness_array)
            population,fitness_array = self.genetic_algorythm(i,population,fitness_array,generations=1,top=1)


    def prepare_vectors(self):
        global arrays
        global population
        global fitness_array

        arrays = {"O": [], "C": [], "E": [], "A": [], "N": []}
        population = {"O": [], "C": [], "E": [], "A": [], "N": []}
        fitness_array = {"O": [], "C": [], "E": [], "A": [], "N": []}

        for i in score_columns:
            # Assuming loadPopulation and loadFitness are functions you've defined
            current_population = self.loadPopulation(i, 500, 40)
            population[i] = current_population
            arrays[i].append(current_population[0, :].reshape(1, -1))

            # If needed, load fitness_array here
            fitness_array[i].append(self.loadFitness(i, current_population))
            # print(f"avg:{i}={np.mean(fitness_array)}")

    def return_X_y(self,index):
        #X=Fitness("O",arrays["O"][0])[:]
        size_x=OCEAN.shape[0]
        X=np.tile(arrays[index][0], (size_x, 1))
        if(index=="O"):
            y=OCEAN[:,:1]
            X=X*self.answers[:size_x,10:]
        elif(index=="C"):
            y=OCEAN[:,1:2]
            X=X*np.delete(self.answers[:size_x,:], np.arange(10, 20),axis=1)
        elif(index=="E"):
            y=OCEAN[:,2:3]
            X=X*np.delete(self.answers[:size_x,:], np.arange(20, 30),axis=1)
        elif(index=="A"):
            y=OCEAN[:,3:4]
            X=X*np.delete(self.answers[:size_x,:], np.arange(30, 40),axis=1)
        elif(index=="N"):
            y=OCEAN[:,4:]
            X=X*np.delete(self.answers[:size_x,:], np.arange(40, 50),axis=1)
        else:
            return None
        return X,y
    
    def addProgress(self):
        for index in score_columns:
            print(f"Vector '{index}' with mse = {fitness_array[index][0][0][0]:.2f}")
            progress_of_optimization_without_answers[index].append(fitness_array[index][0][0][0])

#####################################################
#GENETIC ALGORYTHM POLYONIMAL FUNCTIONS
#####################################################

new_dimension=500 #specifies the size of population

class Genetic_algorythm_Poly_Fitness:
    def __init__(self, OCEAN, answers):
        self.OCEAN = OCEAN
        self.answers=answers

    # Define events for synchronization
    crossover_event = threading.Event()
    mutation_event = threading.Event()

    
    def reset_events(self):
        crossover_event.clear()
        mutation_event.clear()


    logging.basicConfig(level=logging.DEBUG)

    def getSpecificPopulation(self,x=answers.shape[0], y=40, num_ones_per_row=20):
        # Generate an array with zeros
        population = np.zeros((x, y), dtype=int)

        # Set exactly num_ones_per_row ones in each row
        for i in range(x):
            population[i, np.random.choice(y, size=num_ones_per_row, replace=False)] = 1

        return population

    def getPopulation(self,x=answers.shape[0],y=40):
        return np.random.choice([1, 0], size=(x, 40))


        # Generate random indices for ones in each row
        row_indices = np.arange(x)
        #print(row_indices)
        col_indices = np.random.choice(y, size=(x, num_ones_per_row), replace=False)
        #print(col_indices)

        # Set ones in the specified positions
        population[row_indices[:, np.newaxis], col_indices] = 1

        return population

    def loadPopulation(self, index,x=answers.shape[0],y=40):
        #print(f"loadPopulation -> x = answers.shape[0] = {x}, self.answers.shape[0]={self.answers.shape[0]}")
        index_paths = {
            "O": ('DataFiles/population_without_O_poly.csv'),
            "C": ('DataFiles/population_without_C_poly.csv'),
            "E": ('DataFiles/population_without_E_poly.csv'),
            "A": ('DataFiles/population_without_A_poly.csv'),
            "N": ('DataFiles/population_without_N_poly.csv'),
        }
        if os.path.exists(index_paths[index]):
            existing_array = np.loadtxt(index_paths[index], delimiter=',', skiprows=1)
            #print(f" existing_array.shape= {existing_array.shape} , x= {x}")
            if existing_array.shape[0]==x:
                return existing_array
            elif existing_array.shape[0]<x:
                #print("condition=existing_array.shape[0]<x 1")
                x=x-existing_array.shape[0]
                final_array = np.concatenate((existing_array, np.random.choice([1, 0], size=(x, 40))), axis=0)
                print(f"final_array = {final_array.shape} , population.shape={population.shape}")
                return final_array
            elif existing_array.shape[0]>x:
                #print("condition=existing_array.shape[0]>x 2 ")
                final_array = np.random.choice([1, 0], size=(x, 40))
                #final_array= np.ones((x, 40)) # only with 1
                #print(f"final_array = {final_array.shape} , population.shape={population.shape}")
                return final_array
        else:
            return np.random.choice([1, 0], size=(x, 40))

    def savePopulation(self,index,population):
        index_paths = {
        "O": ('DataFiles/population_without_O_poly.csv',answer_columns_without_OPN),
        "C": ('DataFiles/population_without_C_poly.csv',answer_columns_without_CSN),
        "E": ('DataFiles/population_without_E_poly.csv',answer_columns_without_EXT),
        "A": ('DataFiles/population_without_A_poly.csv',answer_columns_without_AGR),
        "N": ('DataFiles/population_without_N_poly.csv',answer_columns_without_NRT),
        }
        path=index_paths[index][0]
        header=index_paths[index][1]
        np.savetxt(path, population, delimiter=',', header=','.join(header), comments='')

    def loadFitness(self,index, population):
        index_paths = {
                "O": ('DataFiles/fitness_array_O_poly.csv'),
                "C": ('DataFiles/fitness_array_C_poly.csv'),
                "E": ('DataFiles/fitness_array_E_poly.csv'),
                "A": ('DataFiles/fitness_array_A_poly.csv'),
                "N": ('DataFiles/fitness_array_N_poly.csv'),
            }
        if os.path.exists(index_paths[index]):
            existing_array = np.loadtxt(index_paths[index], delimiter=',', skiprows=1).reshape(-1,1)
            if existing_array.shape[0]==population.shape[0]:
                return existing_array
            elif population.shape[0]==1:
                final_array = self.Fitness_vector(index,population)
                #print(f"final_array = Fitness_vector(index,population).shape = {final_array.shape}, population.shape= {population.shape}")
                return final_array
            elif population.shape[0]>1:
                final_array = self.Fitness(index,population)
                #print(f"final_array = Fitness(index,population).shape = {final_array.shape}, population.shape= {population.shape}")
                return final_array
        else:
            return self.Fitness(index,population)

    def saveFitness(self,index,fitness_array):
        index_saveFitness = {
                "O": ('DataFiles/fitness_array_O_poly.csv','mse_O'),
                "C": ('DataFiles/fitness_array_C_poly.csv','mse_C'),
                "E": ('DataFiles/fitness_array_E_poly.csv','mse_E'),
                "A": ('DataFiles/fitness_array_A_poly.csv','mse_A'),
                "N": ('DataFiles/fitness_array_N_poly.csv','mse_N'),
        }
        path=index_saveFitness[index][0]
        header=index_saveFitness[index][1]
        np.savetxt(path, fitness_array, delimiter=',', header=header)



    def Fitness_vector(self,index,array1,array2=None):
        array2 = array2 if array2 is not None else self.OCEAN
        #print(f'Fitness_vector =  array2.shape{array2.shape}, OCEAN.shape={OCEAN.shape}')
        array1=array1.reshape(1,-1)
        size_x=OCEAN.shape[0]
        x=array1
        #print(f"Fitness_vector / array1.shape = {array1.shape} ")
        if(array1.shape[1]==40):
            fitness_array = np.empty((size_x, 1))
            x=np.tile(array1, (size_x, 1))
            if(index=="O"):
                y=array2[:size_x,:1]
                x=np.sum(x*self.answers[:size_x,10:],axis=1).reshape(-1,1)
                #print(f"Fitness_vector/if(index=='O'): y.shape = {y.shape} , x.shape={x.shape} ")
            elif(index=="C"):
                y=array2[:size_x,1:2]
                x=np.sum(x*np.delete(self.answers[:size_x,:], np.arange(10, 20), axis=1),axis=1).reshape(-1,1)
            elif(index=="E"):
                y=array2[:size_x,2:3]
                x=np.sum(x*np.delete(self.answers[:size_x,:], np.arange(20, 30), axis=1),axis=1).reshape(-1,1)
            elif(index=="A"):
                y=array2[:size_x,3:4]
                x=np.sum(x*np.delete(self.answers[:size_x,:], np.arange(30, 40), axis=1),axis=1).reshape(-1,1)
            elif(index=="N"):
                y=array2[:size_x,4:]
                x=np.sum(x*np.delete(self.answers[:size_x,:], np.arange(40, 50), axis=1),axis=1).reshape(-1,1)
            else:
                return None
            fitness_vector = self.fitness_poly(x,y)
            #print(f"Fitness_vector / fitness_vector.shape = {fitness_vector.shape} ")
            return fitness_vector

    def perform_fitness(self,x,i,size_x,size_x_population,fitness_array,array1,array2,index):
        x=np.tile(array1[i,:], (size_x, 1))
        if(index=="O"):
            y=array2[:size_x,:1]
            x=np.sum(x*self.answers[:size_x,10:],axis=1).reshape(-1,1)
        elif(index=="C"):
            y=array2[:size_x,1:2]
            x=np.sum(x*np.delete(self.answers[:size_x,:], np.arange(10, 20), axis=1),axis=1).reshape(-1,1)
        elif(index=="E"):
            y=array2[:size_x,2:3]
            x=np.sum(x*np.delete(self.answers[:size_x,:], np.arange(20, 30), axis=1),axis=1).reshape(-1,1)
        elif(index=="A"):
            y=array2[:size_x,3:4]
            x=np.sum(x*np.delete(self.answers[:size_x,:], np.arange(30, 40), axis=1),axis=1).reshape(-1,1)
        elif(index=="N"):
            y=array2[:size_x,4:]
            x=np.sum(x*np.delete(self.answers[:size_x,:], np.arange(40, 50), axis=1),axis=1).reshape(-1,1)
        else:
            return None
        fitness_array[i]=self.fitness_poly(x,y)    

    def Fitness(self,index,array1,array2=None):
        array2 = array2 if array2 is not None else self.OCEAN
        #print(f'Fitness array2.shape{array2.shape}')
        size_x=OCEAN.shape[0]
        size_x_population=array1.shape[0]
        x=array1
        #print(f"array1.shape = {array1.shape}")
        if(array1.shape[0]==10):
            x=x.reshape(1,10) 
            x=np.tile(array1, (OCEAN.shape[0], 1)) 
            if(index=="O"):
                y=array2[:x.shape[0],:1]
                x=x*self.answers[:,:10]
            elif(index=="C"):
                y=array2[:x.shape[0],1:2]
                x=x*self.answers[:,10:20]
            elif(index=="E"):
                y=array2[:x.shape[0],2:3]
                x=x*self.answers[:,20:30]
            elif(index=="A"):
                y=array2[:x.shape[0],3:4]
                x=x*self.answers[:,30:40]
            elif(index=="N"):
                y=array2[:x.shape[0],4:]
                x=x*self.answers[:,40:]
            return self.fitness_poly(x,y)
        elif(array1.shape[0]==1 and array1.shape[1]==40):
            fitness_array = np.empty((size_x, 1))
            threads = []
            for i in range(0,size_x_population):
                thread = threading.Thread(target=self.perform_fitness, args=(x,i,size_x,size_x_population,fitness_array,array1,array2,index))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
        elif(array1.shape[1]==40):
            fitness_array = np.empty((size_x_population, 1))
            threads = []
            for i in range(0,size_x_population):
                thread = threading.Thread(target=self.perform_fitness, args=(x,i,size_x,size_x_population,fitness_array,array1,array2,index))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

        return fitness_array

    def fitness_poly(self,x,y):

        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        # Define the degree of the polynomial
        degree = 2

        # Create polynomial features
        Poly = PolynomialFeatures(degree)
        X_poly_train = Poly.fit_transform(X_train)
        X_poly_test = Poly.transform(X_test)

        # Create and train the Polynomial Regression model
        Poly_reg = LinearRegression()
        Poly_reg.fit(X_poly_train, Y_train)

        # Make predictions on the test set
        Y_poly_pred = Poly_reg.predict(X_poly_test)


        #print_metrics(y1_test, y1_poly_pred)

        # Make predictions on the test set
        #print_metrics(y_test,y_pred)
        #print("x_shape:",x.shape)
        #print("y_shape:",y.shape)    
        #print("y_train:", y_train.shape)
        #print("x_train:", x_train.shape)
        #print("y_test:", y_test.shape)
        #print("x_test:", x_test.shape)

        mse = mean_squared_error(Y_test, Y_poly_pred)
        #print(f"fitness_poly(x,y).shape={mse.shape}")
        return mse

    def sort_arrays(self,population, fitness):
        sorted_indices = np.argsort(fitness, axis=0)
        #print(f"sorted_indices = {sorted_indices}, sorted_indices.shape = {sorted_indices.shape}")
        if(population.shape[0]>1):
            sorted_population = population[sorted_indices[:, 0]]
            sorted_fitness = np.squeeze(fitness[sorted_indices])
        else:
            sorted_population=population
            sorted_fitness = fitness


        return sorted_population, sorted_fitness.reshape(population.shape[0],1)


    def perform_crossover(self,index,population_copy, row, avg, avg2):
        vector=population_copy[row]
        min_value = float('inf')
        for i in range(1,40): #40 must be replaced with population_copy.shape[1]-1
            vector = np.concatenate((population[row, i:], population[row, :i]), axis=0)
            avg2 = self.Fitness_vector(index, population_copy[row])
            if avg2 < min_value and avg2<avg :
                min_value=avg2
                vector = population_copy[row]
        self.crossover_event.set() # Signal that crossover is complete
        if(min_value<avg):
            population_copy[row]=vector
            #logging.debug(f"perform_crossover.success(index, row, min_value, avg)=({index, row, min_value, avg})")
        else:
            pass
            #logging.debug(f"perform_crossover.fail(index, row, min_value, avg)=({index, row, min_value, avg})")


    def crossover(self,index,population, fitness_array):
        x, y = population.shape
        avg = np.mean(fitness_array)
        avg2 = avg
        rows = np.argwhere(fitness_array > avg)[:, 0]
        population_copy = np.copy(population)
        threads = []
        for row in rows:
            thread = threading.Thread(target=self.perform_crossover, args=(index,population_copy, row, avg, avg2))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        self.crossover_event.clear() # Clear the event for the next use   
        return population_copy


    def perform_mutation(self,index, population_copy, row, avg, fitness_array):
        # The original perform_mutation function remains unchanged
        vector = population_copy[row]
        min_value = float('inf')
        for y1 in range(40):
            y2 = np.random.randint(0, 39)
            while y1==y2:
                y2 = np.random.randint(0, 39)
            try:
                new_array = self.swap(population_copy.copy(), row, y1, y2)
                avg2 = self.Fitness_vector(index, new_array[row, :])
                #logging.debug(f"row={row},y1={y1},y2={y2},min_value={min_value}")
                if avg2 < min_value and avg2<avg:
                    min_value = avg2
                    vector = new_array[row]
            except Exception as e:
                logging.error(f"Exception in thread: {e},row={row}, y1={y1},y2={y2},min_value={min_value}")
        if min_value < avg:
            #logging.debug(f"perform_mutation.success(index, row, min_value, avg)=({index, row, min_value, avg})")
            population_copy[row] = vector
        else:
            #logging.debug(f"perform_mutation.fail(y1={y1},y2={y2},min_value={min_value}, index, row, min_value, avg)=({index, row, min_value, avg})")
            pass
        self.mutation_event.set()  # Signal that mutation is complete

    def swap(self,array, row, y1, y2):
        array[row, y1], array[row, y2] = array[row, y2], array[row, y1]
        return array

    def mutation(self,index, population, fitness_array):
        avg = np.mean(fitness_array)
        mut_rows_bad = np.argwhere(fitness_array > avg)[:, 0]
        population_copy = np.copy(population)
        threads = []
        counter = threading.Lock()
        total_rows = len(mut_rows_bad)

        for row in mut_rows_bad:
            thread = threading.Thread(target=self.perform_mutation,
                                      args=(index,
                                            population_copy,
                                            row,
                                            avg,
                                            fitness_array))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        self.mutation_event.clear()  # Clear the event for the next use
        return population_copy



    def genetic_algorythm(self,index,population,fitness_array,generations=1,top=0):
        if(top==0):
            for i in range(generations+1):
                print(f"Generation:{i}, vectors for optimization:",len(np.argwhere(fitness_array > np.mean(fitness_array))[:,0])," avg solution mse:",np.mean(fitness_array))
                print("crossover start")
                population=self.crossover(index,population,fitness_array)
                print("mutation start")
                population=self.mutation(index,population,fitness_array)
                fitness_array=Fitness(index,population)
                population,fitness_array = sort_arrays(population, fitness_array)
        elif(top==1):
            i=0
            avg=np.mean(fitness_array)
            vevtor_for_optimization=len(np.argwhere(fitness_array > avg)[:,0])
            vevtor_for_optimization_prev=population.shape[0]
            condition_of_optimization=vevtor_for_optimization>0
            crossover_loop=False
            mutation_loop=False
            condition_of_loop=crossover_loop and mutation_loop

            print(f"START vevtor_for_optimization prev={vevtor_for_optimization_prev},{vevtor_for_optimization},i={i}")
            population_prev=population.copy()
            population_prev2=population.copy()

            min = avg
            first=avg
            #prev = np.empty((size_x_population, 1))
            while condition_of_optimization and not condition_of_loop:
                start_time = datetime.now()
                print(f"Start Time: {start_time}")
                ###########METRICS############
                print("Generation:",i," vectors for optimization:",vevtor_for_optimization," avg solution mse:",avg)

                i=i+1

                if(not crossover_loop):
                    print("crossover start")
                    population=self.crossover(index,population,fitness_array)
                    crossover_loop=np.array_equal(population, population_prev)
                    population_prev2=self.Fitness(index,population)
                    print(f"crossover end with previous_avg > avg:{np.mean(population_prev2)>np.mean(self.Fitness(index,population_prev))}")
                    print(f"crossover_loop=np.array_equal(population, population_prev)={np.array_equal(population, population_prev)}")
                if(not mutation_loop):
                    print("mutation start")
                    population=self.mutation(index,population,fitness_array)
                    mutation_loop=np.array_equal(population, population_prev)
                    print(f"mutation end with previous_avg > avg:{np.mean(self.Fitness(index,population)) > np.mean(self.Fitness(index,population_prev))}")
                    print(f"mutation_loop=np.array_equal(population, population_prev)={np.array_equal(population, population_prev)}")

                condition_of_loop=crossover_loop and mutation_loop
                fitness_array=self.Fitness(index,population)
                population,fitness_array = self.sort_arrays(population, fitness_array)

                population_prev=population.copy()
                vevtor_for_optimization_prev=vevtor_for_optimization
                prevavg=avg
                avg=np.mean(fitness_array)
                vevtor_for_optimization=len(np.argwhere(fitness_array > avg)[:,0])
                condition_of_optimization=vevtor_for_optimization>0 #and vevtor_for_optimization_prev > vevtor_for_optimization

                print(f"condition_of_optimization={condition_of_optimization} and not condition_of_loop={not condition_of_loop}")
                print(f"before save: condition_of_save (avg<min) = {avg<min}, avg={avg},min={min}")
                ########################SAVE CONTDITION############################
                condition_of_save = avg<min
                if(condition_of_save):
                    self.savePopulation(index,population)
                    self.saveFitness(index,fitness_array)
                    min=avg
                    print("save: ",condition_of_save)
                    print("success:",((prevavg-avg)/(prevavg))*100,"%")
                ########################END SAVE CONTDITION############################




                #population_prev=population.copy()
                #vevtor_for_optimization_prev=vevtor_for_optimization
                #prevavg=avg
                #avg=np.mean(fitness_array)
                #vevtor_for_optimization=len(np.argwhere(fitness_array > avg)[:,0])
                #condition_of_optimization=vevtor_for_optimization>0 #and vevtor_for_optimization_prev > vevtor_for_optimization

                print(f"END vevtor_for_optimization prev={vevtor_for_optimization_prev},new={vevtor_for_optimization},i={i}")


                #########METRICS##############
                #print("condition: ", not np.array_equal(prev, fitness_array))
                end_time = datetime.now()
                # Calculate the time difference
                time_difference = end_time - start_time
                # Print the result
                print(f"End Time: {end_time}")
                print(f"Time Difference: {time_difference}")
                if(i>1000):break
                if(i==10 and first==avg):break

        return population,fitness_array


    def getFunction(self,x,y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        Linreg = LinearRegression()
        Linreg.fit(x_train, y_train)
        y_pred = Linreg.predict(x_test)

        function_coefficients = Linreg.coef_
        function_intercept = Linreg.intercept_

        # Print the function used by the model
        global arrayext
        arrayext=[]
        for i, target_variable in enumerate("O"):
            coefficients = function_coefficients[i]
            arrayext.append(function_coefficients[i])
            terms = [f"{coeff:.2f}*{answer_columns[j]}" for j, coeff in enumerate(coefficients)]
            terms_str = " + ".join(terms)
            function_str = f"Function for O = {function_intercept[i]:.2f} + {terms_str}"
            print(function_str)

    def start_process_of_genetic_algorythm(self):
        global population,fitness_array
        for i in score_columns:
            print_red_bold(f"*********************  genetic algorythm for index {i}  *********************************")
            population=self.loadPopulation(i,new_dimension,40)
            fitness_array=self.loadFitness(i,population)
            #print(f" fitness_array = {fitness_array.shape}")
            population,fitness_array = self.sort_arrays(population, fitness_array)
            population,fitness_array = self.genetic_algorythm(i,population,fitness_array,generations=1,top=1)


    def prepare_vectors(self):
        global arrays_poly
        global population_poly
        global fitness_array_poly

        arrays_poly = {"O": [], "C": [], "E": [], "A": [], "N": []}
        population_poly = {"O": [], "C": [], "E": [], "A": [], "N": []}
        fitness_array_poly = {"O": [], "C": [], "E": [], "A": [], "N": []}

        for i in score_columns:
            # Assuming loadPopulation and loadFitness are functions you've defined
            current_population = self.loadPopulation(i, 500, 40)
            population_poly[i] = current_population
            arrays_poly[i].append(current_population[0, :].reshape(1, -1))

            # If needed, load fitness_array here
            fitness_array_poly[i].append(self.loadFitness(i, current_population))
            # print(f"avg:{i}={np.mean(fitness_array)}")

    def return_X_y(self,index):
        #X=Fitness("O",arrays["O"][0])[:]
        size_x=OCEAN.shape[0]
        X=np.tile(arrays[index][0], (size_x, 1))
        if(index=="O"):
            y=OCEAN[:,:1]
            X=X*self.answers[:size_x,10:]
        elif(index=="C"):
            y=OCEAN[:,1:2]
            X=X*np.delete(self.answers[:size_x,:], np.arange(10, 20),axis=1)
        elif(index=="E"):
            y=OCEAN[:,2:3]
            X=X*np.delete(self.answers[:size_x,:], np.arange(20, 30),axis=1)
        elif(index=="A"):
            y=OCEAN[:,3:4]
            X=X*np.delete(self.answers[:size_x,:], np.arange(30, 40),axis=1)
        elif(index=="N"):
            y=OCEAN[:,4:]
            X=X*np.delete(self.answers[:size_x,:], np.arange(40, 50),axis=1)
        else:
            return None
        return X,y
    
    def addProgress(self):
        for index in score_columns:
            print(f"Vector '{index}' with mse = {fitness_array_poly[index][0][0][0]:.2f}")
            progress_of_optimization_without_answers[index].append(fitness_array_poly[index][0][0][0])

                      
#####################################################
#MACHINE LEARNING IMPORTS
#####################################################
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json


#####################################################
#MACHINE LEARNING FUNCTIONS
#####################################################

activations = {
     1: ('relu', 'Ranges from 0 to positive infinity. Commonly used for hidden layers.'),
     2: ('tanh', 'Ranges from -1 to 1. Often used for hidden layers.'),
     3: ('sigmoid', 'Ranges from 0 to 1. Commonly used in the output layer for binary classification.'),
     4: ('softmax', 'Used in the output layer for multiclass classification. Converts logits to probabilities.'),
     5: ('linear', 'No specific range. Often used in the output layer for regression tasks.'),
     6: ('softplus', 'Ranges from 0 to positive infinity. Smooth approximation of ReLU.'), #36
     7: ('softsign', 'Ranges from -1 to 1. Similar to tanh but with a simpler shape.'),
     8: ('hard_sigmoid', 'Approximates sigmoid but computationally cheaper. Ranges from 0 to 1.'),
     9: ('elu', 'Ranges from negative infinity to positive infinity. A variant of ReLU.'),
     10:('selu', 'Self-normalizing variant of ReLU. Maintains mean and variance during training.'),
     11:('exponential', 'Ranges from 0 to positive infinity. An activation with exponential growth.')
}

min_dimensions=1
max_dimentions=100

class Machine_Learning_Model:
    def __init__(self, genetic_algorythm_class):
        self.genetic_algorythm_class = genetic_algorythm_class

    def remove(self, index):
        index_paths = {
            "O": 'DataFiles/keras_model_for_O',
            "C": 'DataFiles/keras_model_for_C',
            "E": 'DataFiles/keras_model_for_E',
            "A": 'DataFiles/keras_model_for_A',
            "N": 'DataFiles/keras_model_for_N'
        }
        file_path1 = index_paths[index]+ '.json'
        file_path2 = index_paths[index]+ '_weights.h5'

        try:
            os.remove(file_path1)
            os.remove(file_path2)
            print(f"The files '{file_path1},{file_path2}' has been deleted successfully.")
        except FileNotFoundError:
            print(f"The file '{file_path1},{file_path2}' does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def save_model(self, model, index):
        index_paths = {
            "O": 'DataFiles/keras_model_for_O',
            "C": 'DataFiles/keras_model_for_C',
            "E": 'DataFiles/keras_model_for_E',
            "A": 'DataFiles/keras_model_for_A',
            "N": 'DataFiles/keras_model_for_N'
        }

        # File paths
        json_file_path = index_paths[index] + '.json'
        weights_file_path = index_paths[index] + '_weights.h5'

        # Check if files already exist and delete them
        if os.path.exists(json_file_path):
            os.remove(json_file_path)
        if os.path.exists(weights_file_path):
            os.remove(weights_file_path)

        # Save the model architecture to a JSON file
        model_json = model.to_json()
        with open(json_file_path, "w") as json_file:
            json_file.write(model_json)

        # Save the model weights to an HDF5 file
        model.save_weights(weights_file_path)

    def load_model(self, index):

        index_paths = {
            "O": 'DataFiles/keras_model_for_O',
            "C": 'DataFiles/keras_model_for_C',
            "E": 'DataFiles/keras_model_for_E',
            "A": 'DataFiles/keras_model_for_A',
            "N": 'DataFiles/keras_model_for_N'
        }
        if((os.path.exists(index_paths[index] + '.json') and os.path.exists(index_paths[index] + '_weights.h5'))):

            # Load the model architecture from a JSON file
            with open(index_paths[index] + '.json', 'r') as json_file:
                loaded_model_json = json_file.read()

            # Create the model from the loaded architecture
            loaded_model = model_from_json(loaded_model_json)

            # Load the model weights from an HDF5 file
            loaded_model.load_weights(index_paths[index] + '_weights.h5')

            return loaded_model
        else:
            return None

    def accuracy(self, model, index=None, X=None, y=None):
        X_ml,y_ml=self.genetic_algorythm_class.return_X_y(index)
        if not model._is_compiled:
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        if(X.all()==None or y.all()==None):
            mse = model.evaluate(X_ml, y_ml)
        else:
            mse = model.evaluate(X, y)
        return mse


    def generate_model(self,input_dim, output_dim,model=None, activation_function='relu',number_of_units=1):
        print(f"Model: {activation_function}, number_of_units={number_of_units}")
        if(len(model.layers)==0):
            print(len(model.layers))
            #model.pop()
            model.add(Dense(units=number_of_units, input_dim=input_dim, activation=activation_function, name=f'dense_{number_of_units}_{activation_function}_{len(model.layers)+1}st_layer'))
        elif(len(model.layers)>0):
            print(len(model.layers))
            #model.pop()
            model.add(Dense(units=number_of_units, activation=activation_function, name=f'dense_{number_of_units}_{activation_function}_{len(model.layers)+1}nd_layer'))
        print(f"generate_model return with layers ={len(model.layers)}")    
        # Add output layer with linear activation
        #model.add(Dense(units=output_dim, activation='linear'))

        # Compile the model with a random learning rate
        model.compile(optimizer=Adam(learning_rate=np.power(10, np.random.uniform(-4, -2))), loss='mean_squared_error')

        return model

    def ml_search(self, X, y, index):
        global x_data
        global y_data
        x_data = []
        y_data = []
        num_layers = 0
        prev_num_layers=None

        print("Loading Data")
        best_model = self.load_model(index)
        if best_model is not None:
            num_layers = len(best_model.layers)
            best_mse = self.accuracy(best_model, index,X,y)
        else:
            best_model = Sequential()
            #best_model.add(Dense(units=1, input_dim=X.shape[1], activation='relu'))
            best_mse = float('inf')
        mse=0
        generation_model=1
        condition_of_loop_to_continue = prev_num_layers != num_layers
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        while condition_of_loop_to_continue:
            prev_num_layers = num_layers
            #best_model.add(Dense(units=1, activation='relu',name='test'))
            print(f"*******INDEX={index}******generation_model:{generation_model} starting with best_mse: {best_mse} and {len(best_model.layers)} layers**********")
            for key, (activation, description) in activations.items():
                for i in range(1, 3):
                    model = self.generate_model(input_dim=X.shape[1], output_dim=y.shape[1],model=best_model, activation_function=activation, number_of_units=i)
                    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                    # Training loop with while
                    prev_losses = [float('inf'), float('inf'), float('inf')]
                    prev_losses_min = [float('inf'), float('inf'), float('inf'),float('inf'), float('inf'), float('inf')]
                    epoch = 0

                    while True:
                        #print(f"Epoch {epoch + 1}")

                        # Train the model for one epoch
                        history = model.fit(X_train,
                                            y_train,
                                            epochs=1,
                                            batch_size=100,
                                            validation_data=(X_test, y_test),
                                            callbacks=[early_stopping],
                                            verbose=2)

                        # Retrieve the loss value for the current epoch
                        current_loss = history.history['loss'][0]


                        if current_loss >= max(prev_losses) or all(loss != float('inf') and loss > 500 for loss in prev_losses) or all(loss != float('inf') and loss > 120 for loss in prev_losses_min):
                            print(f"Stopping training. Next loss is higher than the {len(prev_losses_min)} previous losses.")
                            break

                        # Update the list of previous losses
                        prev_losses.pop(0)
                        prev_losses.append(current_loss)

                        prev_losses_min.pop(0)
                        prev_losses_min.append(current_loss)

                        epoch += 1

                        mse = model.evaluate(X_test, y_test)
                        if np.isnan(mse):
                            print("NaN loss encountered. Training stopped.")
                            break

                   #metrics
                    y_data.append(mse)
                    x_data.append(generation_model)
                    condition_of_save=mse < best_mse
                    generation_model=generation_model+1
                    if(condition_of_save):
                        clear_output()
                        print_red_bold(f"                              Save:{mse < best_mse}")
                        print_red_bold(f"                              {mse}:new mse")
                        print_red_bold(f"                              {best_mse}:last best_mse")

                        model.summary()
                        best_mse = mse
                        best_model = model
                        self.save_model(model,index)

                        prev_num_layers = num_layers
                        num_layers = len(best_model.layers)

                    else:
                        clear_output()
                        print(f"                              Save:{mse < best_mse}")
                        print(f"                              {mse}:mse")
                        print(f"                              {best_mse}:best_mse")

                        model.pop()


            print(f"final result for index={index}: prev_num_layers={prev_num_layers}, num_layers={num_layers}")
            condition_of_loop_to_continue = prev_num_layers != num_layers              
        return best_model

    def generate_ml_models(self):
        for i in score_columns:
            X_ml, y_ml = self.genetic_algorythm_class.return_X_y(i)
            self.ml_search(X_ml, y_ml, i)

    def prepare_ml_models(self):
        global ml_mse
        global ml_models
        ml_models={"O":[],"C":[],"E":[],"A":[],"N":[]}
        ml_mse={"O":[],"C":[],"E":[],"A":[],"N":[]}
        for i in score_columns:
            X_ml,y_ml=self.genetic_algorythm_class.return_X_y(i)
            best_model = self.load_model(i)
            if best_model is not None:
                ml_models[i].append(best_model)
                model_mse = self.accuracy(best_model, i,X_ml,y_ml)
                progress_of_optimization_without_answers[i].append(model_mse)
                ml_mse[i].append(model_mse)
            else:
                ml_models[i].append(None)
                ml_mse[i].append(None)
        clear_output()

    def show_accuracy_ml_models(self):
        for i in score_columns:
            print(f"model for {i} has mean squared error {ml_mse[i][0]}")










