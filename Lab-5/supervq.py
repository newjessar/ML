import numpy as np
from matplotlib import pyplot as plt
import random

class LabelPoint:
    def __init__(self,feature_vector,label):
        self.feature_vector = feature_vector
        self.label = label


"""This function calculates the minimum distance between a datapoint and all the prototypes
the prototype with the minimum distance is then returned. The distance measured can be varied (maybe in the future)
the default distance measure is euclidian distance"""
def distanceMeasure(prototypes,dataPoint,measure):
    distances = np.zeros(len(prototypes))
    prototypes = np.array(prototypes)
    dataPoint = np.array(dataPoint)
    if measure=="euclid":
        for index,prototype in enumerate(prototypes):
            distances[index]+=np.linalg.norm(prototype-dataPoint,ord=2)
    elif measure=="manhattan":
        for index,prototype in enumerate(prototypes):
            distances[index]+=np.linalg.norm(prototype-dataPoint,ord=1)
    return np.where(distances==np.min(distances))[0][0]

"""
Initialize a set of random prototypes based on the average feature values of the data
"""     
def initialize_random(datas,K):
    data = datas.copy()
    # shuffle the data randomly and put them into groups
    random.shuffle(data)
    newData = [data[i::K] for i in range(K)]
    # set prototypes at 0 value
    prototypes = [[0 for j in range(len(data[0]))] for i in range(K)]
    # go through the data
    for index1 in range(len(newData)):
        for index2 in range(len(newData[index1])):
            for index3 in range(len(newData[index1][index2])):
                # sum the values for the features of the data
                prototypes[index1][index3] += newData[index1][index2][index3]

    # divide the feature values of the prototypes by the amount of data points for which the features were summed
    # so that we have the average instead of the total amount
    for index in range(len(prototypes)):
        for coord in range(len(prototypes[index])):
            prototypes[index][coord] =  prototypes[index][coord] / len(data)
    return prototypes


"""
Make a stupid initialization for the VQ algorithm instead of random
"""
def stupid_init(data,K):
    prototypes = [[20 for j in range(len(data[0]))] for i in range(K)]
    return prototypes
"""
perform the learning algorithm for every epoch
"""
def epoch(data,prototypes,nu,measure):
    shuffledata = data.copy()
    # shuffle the data
    random.shuffle(shuffledata)
    # go through the data
    for indexData in range(len(shuffledata)):
        # find the closest prototype to the data point
        indexPrototype = distanceMeasure(prototypes,shuffledata[indexData],measure)
        # move the prototype closer to the data point
        for index in range(len(prototypes[indexPrototype])):
            prototypes[indexPrototype][index] += nu*(shuffledata[indexData][index]-prototypes[indexPrototype][index])
    return prototypes

"""
Plot the data and prototypes at a certain epoch
"""
def plot_epoch(data,prototypes):
    fig,frame = plt.subplots(1,1)
    prototypes = np.array(prototypes)
    frame.scatter(data[::,0],data[::,1])
    frame.scatter(prototypes[:,0],prototypes[:,1])
    plt.show()

"""
Make a plot of the data and the prototypes but plot the prototypes at different times t with different colors to show
how they moved during the learning process
"""
def plot_trace(trace,data):
    trace = np.array(trace)
    fig,frame = plt.subplots(1,1)
    frame.scatter(data[::,0],data[::,1],s=5)
    for i,prototypes in enumerate(trace):
        frame.scatter(prototypes[:,0],prototypes[:,1],label="step "+str(i+1),s=50)
    frame.set_xlabel("x")
    frame.set_ylabel("y")
    frame.legend()
    plt.show()

"""
Make a plot for the quantization error
"""
def plot_HQ(HQ):
    fig,frame = plt.subplots(1,1)
    frame.plot(range(len(HQ)),HQ)
    frame.set_title("HQ")
    frame.set_xlabel("time t (epochs)")
    frame.set_ylabel("quantization error")
    plt.show()

"""
Make multiple plots for the quantization error for multiple runs of the learning rate nu and number 
of prototypes K
"""
def multi_plot_HQ(HQ,nus,Ks):
    fig,frames = plt.subplots(len(nus),len(Ks),figsize=(8,8))
    for i in range(len(nus)):
        for j in range(len(Ks)):
            index = i*len(Ks)+j
            print(index)
            frames[i,j].plot(range(len(HQ[index])),HQ[index])
            frames[i,j].set_title("nu: "+str(nus[i])+" K: "+str(Ks[j]))
            frames[i,j].set_xlabel("time t (epochs)")
            frames[i,j].set_ylabel("quantization error")
    fig.tight_layout()
    plt.show()
    
"""
evaluates what the quantization error is of the prototypes wrt the data
"""
def evaluate_HQerror(data,prototypes,measure):
    HQ = 0
    # go through the data
    for point in data:
        # find the closest prototype to a data point called point
        indexPrototype = distanceMeasure(prototypes,point,measure)
        # calculate the squared distance from the prototype to the data point
        if measure=="euclid":
            HQ += np.linalg.norm(point-prototypes[indexPrototype],ord=2)**2
        elif measure=="manhattan":
            HQ += np.linalg.norm(point-prototypes[indexPrototype],ord=2)**2
    return HQ

"""
Perform the full algorithm and make plots
"""
def VQlearning(data,K,tmax,nu,measure="euclid"):
    prototypes = initialize_random(data,K)
    HQlist = []
    t=1
    trace = []
    trace.append([])
    for i in range(len(prototypes)):
        trace[-1].append([])
        for j in prototypes[i]:
            trace[-1][i].append(j)
    # perform the learning algorithm for one epoch until t is higher than tmax
    while t<=tmax:
        # perform the learning
        prototypes = epoch(data,prototypes,nu,measure)
        # evaluate the quantization error
        HQ = evaluate_HQerror(data,prototypes,measure)
        HQlist.append(HQ)
        # at certain moments save the location of the prototypes
        if (t%20 ==0):
            trace.append([])
            for i in range(len(prototypes)):
                trace[-1].append([])
                for j in prototypes[i]:
                    trace[-1][i].append(j)
            # plot_epoch(data,prototypes)
        t += 1
    # plot the trace and the quantization error in seperate plots
    #plot_trace(trace,data)
    #plot_HQ(HQlist)
    return prototypes,HQlist

def main():
    data = np.loadtxt("simplevqdata.csv",delimiter = ",")
    nu = [0.1,0.4,0.7]
    K = [2,4]
    prototypes,HQlist = VQlearning(data,4,100,0.05)
    moreHQlist = []
    for i in nu:
        for j in K:
            prototypes,HQlist = VQlearning(data,j,100,i)
            moreHQlist.append(HQlist)
    multi_plot_HQ(moreHQlist,nu,K)
main()
    
    