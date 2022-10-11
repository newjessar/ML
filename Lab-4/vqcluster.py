import numpy as np
from matplotlib import pyplot as plt
import random

"""This function calculates the minimum distance between a datapoint and all the prototypes
the prototype with the minimum distance is then returned. The distance measured can be varied (maybe in the future)
the default distance measure is euclidian distance"""
def distanceMeasure(prototypes,dataPoint,measure="euclid"):
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
            
def initialize_random(data,K):
    #print(data)
    random.shuffle(data)
    newData = [data[i::K] for i in range(K)]
    #print(newData)
    prototypes = [[0 for j in range(len(data[0]))] for i in range(K)]
    for index1 in range(len(newData)):
        for index2 in range(len(newData[index1])):
            #print(prototypes[index1][index2],newData[index1][index2])
            for index3 in range(len(newData[index1][index2])):
                prototypes[index1][index3] += newData[index1][index2][index3]

    for index in range(len(prototypes)):
        for coord in range(len(prototypes[index])):
            prototypes[index][coord] =  prototypes[index][coord] / len(data)
    return prototypes

def epoch(data,prototypes,nu):
    shuffledata = data.copy()
    random.shuffle(shuffledata)
    for indexData in range(len(shuffledata)):
        indexPrototype = distanceMeasure(prototypes,shuffledata[indexData])
        for index in range(len(prototypes[indexPrototype])):
            prototypes[indexPrototype][index] += nu*(shuffledata[indexData][index]-prototypes[indexPrototype][index])
    return prototypes

def plot_epoch(data,prototypes):
    fig,frame = plt.subplots(1,1)
    prototypes = np.array(prototypes)
    frame.scatter(data[::,0],data[::,1])
    frame.scatter(prototypes[:,0],prototypes[:,1])
    plt.show()

def plot_HQ(HQ):
    fig,frame = plt.subplots(1,1)
    frame.scatter(range(len(HQ)),HQ)
    frame.set_title("HQ")
    plt.show()
    
def evaluate_HQerror(data,prototypes,measure="euclid"):
    HQ = 0
    for point in data:
        indexPrototype = distanceMeasure(prototypes,point,measure)
        if measure=="euclid":
            HQ += np.linalg.norm(point-prototypes[indexPrototype],ord=2)**2
        elif measure=="manhattan":
            HQ += np.linalg.norm(point-prototypes[indexPrototype],ord=2)**2
    return HQ


def VQlearning(data,K,tmax,nu):
    prototypes = initialize_random(data,K)
    HQlist = []
    t=1
    #print(prototypes)
    while t<=tmax:
        #print(data)
        prototypes = epoch(data,prototypes,nu)
        HQ = evaluate_HQerror(data,prototypes)
        HQlist.append(HQ)
        if (t%50 ==0):
            t += 0
            plot_epoch(data,prototypes)
            #print(prototypes)
        t += 1
    plot_HQ(HQlist)


def main():
    data = np.loadtxt("simplevqdata.csv",delimiter = ",")
    P = len(data)
    N = len(data[0])
    print(data)
    VQlearning(data,2,100,0.1)
    
main()
    
    