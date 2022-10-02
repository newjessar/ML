from sqlite3 import dbapi2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from numpy import diff

def regionQuery(data,point,epsilon):
    N = []
    for i,x in enumerate(data):
        if (point[0]-x[0])**2+(point[1]-x[1])**2 < epsilon:
            N.append(i)
    return N
        
def expandCluster(P,indexList,data,neighbourPts,epsilon,minPts,clusters):
    c = []
    c.append(P)
    indexList[P]=1
    #neigbourPts3 = []
    for i in neighbourPts:
        if indexList[i]:
            continue
        else:
            indexList[i]=1
            neighbourPts2 = regionQuery(data,data[i],epsilon)
            if len(neighbourPts2) > minPts:
                neighbourPts += neighbourPts2
        flag = 1
        for j in list(np.concatenate(clusters).flat):
            if (j==i):
                flag = 0
        if flag:
            c.append(i)
    return c,indexList


def DBSCAN(data,epsilon,minPts):
    indexList = np.zeros(len(data))
    noise = []
    C = []
    for item in range(len(data)):
        if (indexList[item]):
            continue
        else:
            indexList[item]=1
        neighborPts = regionQuery(data,data[item],epsilon)
        if (len(neighborPts)< minPts):
            noise.append(item)
        else:
            c,indexList = expandCluster(item,indexList,data,neighborPts,epsilon,minPts)
            C.append(c)
    return C
        
def knearneigbors(data,valuess):
    neibList = []
    kneibList = []
    distances = []
    for item in range(len(valuess)):
        neibList.append(NearestNeighbors(n_neighbors = valuess[item]))
        neibList[item].fit(data)
        kneibList.append(neibList[item].kneighbors(data, return_distance=True)[0])

        distances.append([])
        for item2 in kneibList[item]:
            #print(item2)
            distances[item].append(item2[-1])
        #print(distances[item])
        distances[item] = sorted(distances[item])

    return distances


def derv(y):
    dy = diff(y)
    dx = diff(range(len(y)))
    dydx = dy/dx
    ddyddx = diff(dydx)/diff(dx)
    fig = plt.figure()
    frame = fig.add_subplot(1,1,1)
    print(ddyddx)
    frame.plot(diff(dx),ddyddx)   
    plt.show()

def plot_knearneigbors(distances):
    fig = plt.figure()
    frame = fig.add_subplot(1,1,1)
    
    for item in distances:
        frame.plot(range(len(item)), item)

    plt.show()

def main():
    data = np.loadtxt("data_clustering.csv",delimiter = ",")
    valuess = np.array([3, 4, 5])
    distances = knearneigbors(data,valuess)
    derv(distances[0])
    #plot_knearneigbors(distances)

    #x = DBSCAN(data,)
    #print(x)

if __name__ == "__main__":
    main()