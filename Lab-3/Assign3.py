from sqlite3 import dbapi2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from numpy import diff

def regionQuery(data,point,epsilon):
    N = []
    for i,x in enumerate(data):
        if ((point[0]-x[0])**2+(point[1]-x[1])**2)**(1/2) < epsilon:
            N.append(i)
    return N

def find_elbow(data, theta):

    # make rotation matrix
    co = np.cos(theta)
    si = np.sin(theta)
    rotation_matrix = np.array(((co, -si), (si, co)))

    # rotate data vector
    rotated_vector = data.dot(rotation_matrix)

    # return index of elbow
    return np.where(rotated_vector == rotated_vector[:, 1].min())[0][0]

def get_data_radiant(data):
  return np.arctan2(data[:, 1].max() - data[:, 1].min(), 
                    data[:, 0].max() - data[:, 0].min())
        
def expandCluster(P,indexList,data,neighbourPts,epsilon,minPts,clusters):
    c = []
    c.append(P)
    indexList[P]=1
    clusters2 = clusters.copy()
    #neigbourPts3 = []
    for i in neighbourPts:
        if indexList[i]==0:
            indexList[i]=1
            neighbourPts2 = regionQuery(data,data[i],epsilon)
            if len(neighbourPts2) > minPts:
                for j in neighbourPts2:
                    neighbourPts.append(j)
        flag = 1
        clus = []
        for cluster in clusters2:
            if type(cluster) != list:
                clus.append(cluster)
                continue
            for j in cluster:
                clus.append(j)
        #print(clus)
        for j in clus:
            if (j==i):
                flag = 0
        if flag:
            clusters2.append(i)
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
            if (len(neighborPts) < minPts):
                noise.append(item)
            else:
                print("test")
                c,indexList = expandCluster(item,indexList,data,neighborPts,epsilon,minPts,C)
                C.append(c)
    return C
        
def knearneigbors(data,valuess):
    neibList = []
    kneibList = []
    distances = []
    # go through the list valuess which is a list of all the k neighbours
    # 
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


def plot_knearneigbors(distances,labels):
    fig = plt.figure()
    frame = fig.add_subplot(1,1,1)
    
    for item in range(len(distances)):
        frame.plot(range(len(distances[item])), distances[item],label=str(labels[item]))
    frame.legend()
    plt.show()

def findEpsilon(distances):
    epsilons = []
    for dis in distances:
        elbow = []
        for i,x in enumerate(dis):
            elbow.append([i,x])
        elbow = np.array(elbow)
        #print(elbow)
        index = find_elbow(elbow, get_data_radiant(elbow))
        epsilons.append(dis[index])
    return epsilons

def main():
    data = np.loadtxt("Lab-3/data_clustering.csv",delimiter = ",")
    valuess = np.array([3, 4, 5])
    distances = knearneigbors(data,valuess)
    #derv(distances[0])
    plot_knearneigbors(distances,valuess)

    epsilons = findEpsilon(distances)
    print(epsilons)
    x = DBSCAN(data,epsilons[2],5)
    print(x)

if __name__ == "__main__":
    main()