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
    #print("datapoint: ", dataPoint)
    if measure=="euclid":
        for index,prototype in enumerate(prototypes):
            #print(prototype.feature_vector)
            distances[index]+=np.linalg.norm(prototype.feature_vector-dataPoint.feature_vector,ord=2)
    elif measure=="manhattan":
        for index,prototype in enumerate(prototypes):
            distances[index]+=np.linalg.norm(prototype.feature_vector-dataPoint.feature_vector,ord=1)
    return np.where(distances==np.min(distances))[0][0]

"""
Initialize a set of random prototypes based on the average feature values of the data
"""     
def initialize_random_partition(datas,K,prototypes):
    data = datas.copy()
    # shuffle the data randomly and put them into groups
    random.shuffle(data)
    newData = [data[i::K] for i in range(K)]
    # go through the data
    for index1 in range(len(newData)):
        for index2 in range(len(newData[index1].feature_vector)):
            label = newData[index1][index2].label
            for index3 in range(len(newData[index1][index2].feature_vector)):
                # sum the values for the features of the data
                prototypes[index1].feature_vector[index3] += newData[index1][index2].feature_vector[index3]

    # divide the feature values of the prototypes by the amount of data points for which the features were summed
    # so that we have the average instead of the total amount
    for index in range(len(prototypes)):
        for coord in range(len(prototypes[index])):
            prototypes[index].feature_vector[coord] =  prototypes[index].feature_vector[coord] / len(data)
    return prototypes

def sameVector(prototype,point):
    for feature in range(len(point)):
        if (point[feature]!=prototype[feature]):
            return False
    return True

def used(point,prototypes):
    for prototype in prototypes:
        if sameVector(prototype.feature_vector,point.feature_vector):
            return True
    return False

def initialize_random_single(data, prototypes):
    newData = data.copy()
    random.shuffle(newData)
    for prototype in prototypes:
        for point in newData:
            if point.label==prototype.label and not used(point,prototypes):
                for index,feature in enumerate(point.feature_vector):
                    prototype.feature_vector[index] = feature.copy()
    return prototypes                       
        
    
"""
perform the learning algorithm for every epoch
"""
def epoch(data,prototypes,eta,measure):
    shuffledata = data.copy()
    # shuffle the data
    random.shuffle(shuffledata)
    # go through the data
    for indexData in range(len(shuffledata)):
        # find the closest prototype to the data point
        indexPrototype = distanceMeasure(prototypes,shuffledata[indexData],measure)
        if (prototypes[indexPrototype].label==shuffledata[indexData].label):
            same = 1
        else:
            same = -1
        # move the prototype closer to the data point
        for index in range(len(prototypes[indexPrototype].feature_vector)):
            prototypes[indexPrototype].feature_vector[index] += \
                same*eta*(shuffledata[indexData].\
                feature_vector[index]-prototypes[indexPrototype].feature_vector[index])
    return prototypes

"""
Make a plot of the data and the prototypes but plot the prototypes at different 
times t with different colors to show
how they moved during the learning process
"""
def plot_trace(trace,data,n_class):
    fig,frame = plt.subplots(1,1)
    marking = ["o","s"]
    color = ["red","green","blue","yellow","pink","black"]
    print(data)
    for i in range(n_class):
        frame.scatter(data[50*i:50*(i+1):,0],\
        data[50*i:50*(i+1):,1],s=5,marker=marking[i])
    for i,prototypes in enumerate(trace):
        clusters = [[] for i in range(n_class)]
        for prototypeIndex in range(len(prototypes)):
            clusters[prototypes[prototypeIndex].label].\
            append(prototypes[prototypeIndex].feature_vector)
        clusters = np.array(clusters)
        for j in range(n_class):
            frame.scatter(clusters[j][:,0],clusters[j][:,1],marker=marking[j],\
            label="step "+str(i+1) + " cat "+ str(j),s=50,c=color[i])
    frame.set_xlabel("x")
    frame.set_ylabel("y")
    frame.legend()
    plt.show()


"""
first put the labels on all data
"""
def label_data(data,K,n_class):
    dim = len(data[0])
    prototypes = []
    for i in range(n_class):
        for j in range(K):
            features = [0 for k in range(dim)]
            prototypes.append(LabelPoint(features,i))
    newData = []
    for i,x in enumerate(data):
        newData.append(LabelPoint(x,i//(len(data)/n_class)))
    return prototypes,newData

"""
This function calculated the training error by looking if the classification based on the prototypes matches the actual
label of the data. The training 
"""
def calc_training_error(data,prototypes,measure):
    E = 0
    for dataPoint in data:
        indexPrototype = distanceMeasure(prototypes,dataPoint,measure)
        if (dataPoint.label!= prototypes[indexPrototype].label):
            E +=1
    return E

"""
This function plots the training error over a range of epochs by just taking the indexes of the list of training errors
"""
def plot_training_error(E):
    E = np.array(E)
    E = E/100
    fig,frame = plt.subplots(1,1)
    frame.plot(range(len(E)),E)
    frame.set_xlabel("epoch")
    frame.set_ylabel("Training error")
    plt.show()



"""
Perform the full algorithm and make plots
"""
def VQlearning(data,K,tmax,eta,n_class=2,measure="euclid"):
    prototypes,labelData = label_data(data,K,n_class)
    prototypes = initialize_random_single(labelData,prototypes)
    t=1
    trainE = []
    trace = []
    trace.append([])
    for i in range(len(prototypes)):
        feat = []
        for j in range(len(prototypes[i].feature_vector)):
            feat.append(prototypes[i].feature_vector[j])
        prototypecopy = LabelPoint(feat,prototypes[i].label)
        trace[-1].append(prototypecopy)
    # perform the learning algorithm for one epoch until t is higher than tmax
    while t<=tmax:
        # perform the learning
        prototypes = epoch(labelData,prototypes,eta,measure)
        # calculate the training error
        E = calc_training_error(labelData,prototypes,measure)
        trainE.append(E)

        # check if the training error is constant if so stop running early
        if (len(trainE)>50):
            flag = True
            for i in range(50):
                if abs(trainE[t-1]-trainE[t-i-1]) > 0:
                    flag = False
            if flag:
                # add one more last trace for the prototype positions
                trace.append([])
                for i in range(len(prototypes)):
                    feat = []
                    for j in range(len(prototypes[i].feature_vector)):
                        feat.append(prototypes[i].feature_vector[j])
                    prototypecopy = LabelPoint(feat,prototypes[i].label)
                    trace[-1].append(prototypecopy)
                break

        # at certain moments save the location of the prototypes
        if (t%(tmax//5)==0):
            trace.append([])
            for i in range(len(prototypes)):
                feat = []
                for j in range(len(prototypes[i].feature_vector)):
                    feat.append(prototypes[i].feature_vector[j])
                prototypecopy = LabelPoint(feat,prototypes[i].label)
                trace[-1].append(prototypecopy)
        t += 1
    # plot the trace and the classification error in seperate plots
    plot_training_error(trainE)
    plot_trace(trace,data,n_class)
    return prototypes

def main():
    data = np.loadtxt("/Users/newjessar/Documents/GitHub/ML-RUG/Lab-5/lvqdata.csv",delimiter = ",")
    prototypes = VQlearning(data,1,400,0.002)
    
main()
    
    