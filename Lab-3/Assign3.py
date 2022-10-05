from operator import index
from sqlite3 import dbapi2
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from numpy import diff

# look which points are within the neigborhood of a point called point, the distance is euclidian distance
# this includes the point itself. return all the neigbor points as a list of indexes.
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

# look at one point and look at all the points that are either directly or indirectly connect to that point
# all these points are either put into the same cluster or they were classified before and then nothing is done
def expandCluster(P,indexList,data,neighbourPts,epsilon,minPts,clusters):
    c = []
    c.append(P)
    indexList[P]=1
    clusters2 = clusters.copy()

    # go through all the neigbour points and look at the neigbors of every neigbor point and add that to all
    # neigborhood points
    for i in neighbourPts:
        if indexList[i]==0:
            indexList[i]=1
            neighbourPts2 = regionQuery(data,data[i],epsilon)
            if len(neighbourPts2) > minPts:
                for j in neighbourPts2:
                    neighbourPts.append(j)

        # set a flag for later
        flag = 1
        clus = []

        # make a flat list of all the points that are already classified in a cluster
        for cluster in clusters2:
            if type(cluster) != list:
                clus.append(cluster)
                continue
            for j in cluster:
                clus.append(j)

        # go through the clus list and check if the current point index is already in that list
        # if so do nothing else append the current point to the cluster list
        for j in clus:
            if (j==i):
                flag = 0
        if flag:
            clusters2.append(i)
            c.append(i)
    return c,indexList

# perform the DBSCAN algorithm using the functions regionQuery to determine all the points that are in the neigborhood of a point
# and expandcluster to determine all the points from the same cluster
# and also in the process determine which points are noice
def DBSCAN(data,epsilon,minPts):
    indexList = np.zeros(len(data))
    noise = []
    C = []
    for item in range(len(data)):
        # if the point is already visited do nothing
        if (indexList[item]):
            continue
        else:
            indexList[item]=1
            neighborPts = regionQuery(data,data[item],epsilon)
            # if there are less neigborhood points than the minimum requirement then classify it as noise.
            if (len(neighborPts) < minPts):
                noise.append(item)
            else:
                c,indexList = expandCluster(item,indexList,data,neighborPts,epsilon,minPts,C)
                C.append(c)
    return C,noise

# determine the distance from each point to its farthest neigbour as determined from the NearestNeigbors
# function from sklearn 
def knearneigbors(data,valuess):
    neibList = []
    kneibList = []
    distances = []
    # go through the list valuess which is a list of all the k neighbours
    for item in range(len(valuess)):
        neibList.append(NearestNeighbors(n_neighbors = valuess[item]))
        neibList[item].fit(data)
        kneibList.append(neibList[item].kneighbors(data, return_distance=True)[0])

        # put the distances for each n_neigbor value in a list in side of a list. and sort the list
        distances.append([])
        for item2 in kneibList[item]:
            distances[item].append(item2[-1])
        distances[item] = sorted(distances[item])

    return distances

# we make a plot of the distances we got from the knearneigbours function
# we plot this for visualisation to help us in understanding if the value for epsilon makes sense
def plot_knearneigbors(distances,labels):
    fig = plt.figure()
    frame = fig.add_subplot(1,1,1)
    
    for item in range(len(distances)):
        frame.plot(range(len(distances[item])), distances[item],label=str(labels[item]))

    frame.set_xlabel("index")
    frame.set_ylabel("distance")
    frame.legend()
    plt.show()

# we try to find the optimal epsilon value for every
# number of minpoints we use a function for this called find_elbow
def findEpsilon(distances):
    epsilons = []
    for dis in distances:
        elbow = []
        for i,x in enumerate(dis):
            elbow.append([i,x])
        elbow = np.array(elbow)
        index = find_elbow(elbow, get_data_radiant(elbow))
        epsilons.append(dis[index])
    return epsilons

# make plots of the clusters
def plot_clusters(clusters,data,noise):
    fig,frames = plt.subplots(1,1,figsize=(10,10))

    # plot the clusters
    for i in range(len(clusters)):
        frames.scatter(data[clusters[i],0],data[clusters[i],1])
    # plot the noise
    frames.scatter(data[noise,0],data[noise,1],label="noise")
    frames.legend()
    frames.set_xlabel("x")
    frames.set_ylabel("y")
    frames.set_title("DBSCAN clustering algorithm")

    plt.show()

# calculate the silhouette scores for the clustering according to indexesold which are just
# the indexes for the data points clusters
def silDistance(data,indexesold):
    # first make a flat list from indexesold so that we can index the data because we do not use the noise
    indexes = []
    for i in indexesold:
        for j in i:
            indexes.append(j)
    n_clusters = len(indexesold)

    # next make a list of labels to label each cluster with a number
    labels = []
    q = -1
    for i in indexesold:
        q+=1
        for j in i:
            labels.append(q)

    # and lastly calculate the silhouette scores by looping through the data
    Sx = 0
    for i,x in enumerate(data[indexes]):
        clusters = []
        for m in range(n_clusters):
            clusters.append([m,0,0])
        ax = 0
        bx = 0
        dis = 0
        Ci = 0
        for j,y in enumerate(data[indexes]):
            if i==j:
                Ci += 1
                continue
            if (labels[j]==labels[i]):
                dis = dis + (x[0]-y[0])**2+(x[1]-y[1])**2
                Ci += 1
                continue
            for k in clusters:
                if labels[j]== k[0]:
                    k[1] += (x[0]-y[0])**2+(x[1]-y[1])**2
                    k[2] += 1
        bxx = []
        for l in clusters:
            if (l[0]==labels[i]):
                continue
            bxx.append(l[1]/l[2])
        bx = min(bxx)
        ax = dis/Ci
        Sx += (bx-ax)/max([ax,bx])

    # return the average silhouette score
    return Sx/len(data)
        

def main():
    data = np.loadtxt("data_clustering.csv",delimiter = ",")
    valuess = np.array([3, 4, 5])
    distances = knearneigbors(data,valuess)
    plot_knearneigbors(distances,valuess)

    epsilons = findEpsilon(distances)
    print(epsilons)
    # calculate the silhouette scores
    for i in range(3):
        clusters,noise = DBSCAN(data,epsilons[i],i+3)

        Sx = silDistance(data,clusters)
        print(Sx)

    plot_clusters(clusters,data,noise)
    

if __name__ == "__main__":
    main()