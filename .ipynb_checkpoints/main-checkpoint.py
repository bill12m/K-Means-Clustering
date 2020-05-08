import pandas as pd
import numpy as np
import subprocess as sp
from numpy.random import default_rng
import matplotlib.pyplot as plt

sp.call('clear', shell = True)
#Functions for the proximity measurements
def square_error(x,mean):
    error = 0
    for dimension in range(len(x)):
        error += np.square(x[dimension] - mean[dimension])
    
    error = np.square(np.sqrt(error))
    return error
def euclidean_distance(x,mean):
    error = 0
    for dimension in range(len(x)):
        error += np.square(x[dimension] - mean[dimension])
        
    error = np.sqrt(error)
    return error

cluster = [2,4]

for k in cluster:
#I ran the line of code below once to generate my seeds. I then copied the
#results and commented it out. I left it here for you to see where the
##seeds came from.
    #seeds = rng.integers(low = 1, high = 100, size = 10)
    seeds = np.asarray([55,7,72,38,57,43,47,65,68,12])
    for seed in seeds:
        #Construct a random number generator for this seed and construct a
        #training set with it.
        rng = default_rng(seed)
    
        data_train = pd.DataFrame(
            np.zeros(shape = (50,3)), columns = ['x','y','Cluster'])
        for index in data_train.index:
            data_train.loc[index,'x'] = rng.uniform(low = 1,high = 100,size = 1)
            data_train.loc[index,'y'] = rng.uniform(low = 1,high = 100,size = 1)
    
        ##Create dataframe to contain errors##
            
        #Construct multi-level index
        index_array = [[None]*2*k for _ in range(2)]
        for i in range(1, k + 1):
            index_array[0][2 * i - 2] = 'Cluster '+str(i)
            index_array[0][2 * i - 1] = 'Cluster '+str(i)
            if i % 2 == 1:
                index_array[1][i - 1] = index_array[1][i + 1] = 'Intra-sq-dist'
            else:
                index_array[1][i - 1] = index_array[1][i + 1] = 'Intra-dist'
            if i % 4 == 0:
                index_array[1][6] = 'Intra-sq-dist'
                index_array[1][7] = 'Intra-dist'
        tuples = list(zip(*index_array))
        errors_index = pd.MultiIndex.from_tuples(tuples,names = ['first','second'])
        #Construct an empty dataframe with the index we created
        errors = pd.DataFrame(np.zeros(shape = (100,2*k)),
                              columns = errors_index)
        del(errors_index,tuples,index_array,i)
        errors['TSSE'] = errors['TSE'] = np.zeros(shape = len(errors))
        #General variables we'll need for the algorithm
        means = []
        iterations = 0
        print("Seed: ", seed)
        
        #Initialize means 
        means = np.asarray(
            data_train[['x','y']].sample(
                n = k,replace = False,random_state = seed))
        
        while iterations < 100:
            #Label each point to its correct cluster
            for index in data_train.index:
                minimal_distance = 200
                for mean in means:
                    distance = euclidean_distance(
                        x = np.asarray(data_train.iloc[index,0:2]),
                        mean = mean)
                    if distance < minimal_distance:
                        minimal_distance = distance
                        data_train.loc[index,'Cluster'] = mean[0]
            data_train = data_train.sort_values(by = 'Cluster').reset_index()
            del(distance, minimal_distance, data_train['index'])
            
            #Calculate the error for a single cluster
            for mean in means:
                sum_square_error = 0
                sum_error = 0
                
                for index in data_train.index:
                    if data_train.loc[index,'Cluster'] == mean[0]:
                        sum_square_error += square_error(
                            x = np.asarray(data_train.iloc[index,0:2]),
                            mean = mean)
                        sum_error += euclidean_distance(
                            x = np.asarray(data_train.iloc[index,0:2]),
                            mean = mean)
                errors_index = np.argwhere(means[:,0] == mean[0])
                #Store errors for each cluster
                errors.loc[iterations,('Cluster '+str(errors_index[0,0] + 1),'Intra-sq-dist')] = sum_square_error
                errors.loc[iterations,('Cluster '+str(errors_index[0,0] + 1),'Intra-dist')] = sum_error
            #Calculate the total errors over this iteration
            errors.loc[iterations,'TSSE'] = errors.xs('Intra-sq-dist',axis = 1, level = 'second').loc[iterations,:].sum()
            errors.loc[iterations,'TSE'] = errors.xs('Intra-dist',axis = 1, level = 'second').loc[iterations,:].sum()
            del(errors_index)
            
            #Updata the mean for each cluster
            new_means = []
            for mean in means:
                individual_cluster = data_train[data_train['Cluster'] == mean[0]]
                new_means.append(np.asarray(individual_cluster.loc[:,['x','y']].mean()))
            #Stopping condition to check if the means have not changed over
            #this iteration
            new_means = np.asarray(new_means)
            if np.allclose(means,new_means) is True:
                break
            
            means = new_means
            iterations +=1

        #If the stopping condition was met, find the rows of our 'errors'
        #dataframe which were never used and remove them.
        errors = errors.where(errors > 0).dropna()
        
        #Check the 'errors' dataframe for iterations where the error
        #increased        
        #change = errors.pct_change()
        #for index in change.index:
        #    for column in change.columns:
        #        if change.loc[index,column] > 0:
        #            errors.apply(
        #                lambda x: 'lightgreen')
        #            errors.applymap(
        #                lambda x: 'green')
        
        #Print 1 sample dataframe or errors for each cluster
        if seed == 55:
            print(errors)
            color = data_train['Cluster']
            plt.scatter(data_train['x'],data_train['y'], c = color, cmap = 'Set1')
            plt.show()
