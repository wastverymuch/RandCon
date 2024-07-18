

import numpy as np
import pandas as pd

def coupling(data,window):
    """
        creates a functional coupling metric from 'data'
        data: should be organized in 'time x nodes' matrix
        smooth: smoothing parameter for dynamic coupling score
    """
    
    #define variables
    [tr,nodes] = data.shape
    der = tr-1
    td = np.zeros((der,nodes))
    td_std = np.zeros((der,nodes))
    data_std = np.zeros(nodes)
    mtd = np.zeros((der,nodes,nodes))
    sma = np.zeros((der,nodes*nodes))
    
    #calculate temporal derivative
    for i in range(0,nodes):
        for t in range(0,der):
            td[t,i] = data[t+1,i] - data[t,i]
    
    
    #standardize data
    for i in range(0,nodes):
        data_std[i] = np.std(td[:,i])
    
    td_std = td / data_std
   
   
    #functional coupling score
    for t in range(0,der):
        for i in range(0,nodes):
            for j in range(0,nodes):
                mtd[t,i,j] = td_std[t,i] * td_std[t,j]


    #temporal smoothing
    temp = pd.DataFrame(np.reshape(mtd,[der,nodes*nodes]))
    sma = temp.rolling(window).mean()
    sma = np.reshape(sma.to_numpy(),[der,nodes,nodes])
    
    return (mtd, sma)
    
    

#input the variables 'd' (data) and 's' (smooth) 
#I've made 'd' random for now, but this could just as easily be real data
d = np.random.rand(200,5)
s = 9

#run the script
x = coupling(d,s)
pass