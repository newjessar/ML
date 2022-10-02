import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io
import pandas as pn
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

def DBSCAN (data):
    C = 0
    clusterArr = [data[0]]
    indexArr = []
    for item in range(1, len(data)):
        p1 = data[item-1]
        p2 = data[item]
        print(p1)
        break
            
            
    return 0


# def regionQuery(Data, P, eps)

def main():
     
 
    # Loading the data
    file = open("Lab-3/data_clustering.csv", 'rb')
    data = np.loadtxt(file,delimiter = ",")
    
    fig = plt.figure()
    frame = fig.add_subplot(1,1,1)
    
    valuess = np.array([3, 4, 5])
    neibList = []
    kneibList = []
    distances = []
    for item in range(lem(valuess)):
        neibList.append(NearestNeighbors(n_neighbours = valuess[item]))
        neibList[item].fit(data)
        kneibList.append(neibList[item].kneighbors(data, return_distance=True)[0])

        distances = []
        for item2 in kneibList[item]:
            distances[item].append(item2[-1])
        sorted(distances[item])

        plt.plot(range(len(distances[item])), distances[item])

        
    plt.show()
    

    
    
    
    
    #    def get_clusters(self, data, cluster_threshold=0.05):
    #     clusterFinal = [] 
    # if data:   
    #   clusterArr = [data[0]]
    #   for item in range(1, len(data)):
    #         p1 = data[item-1]
    #         p2 = data[item]
    #         pointsDis = self.distance(p1, p2)
    #         if pointsDis <= cluster_threshold:
    #             clusterArr.append(p2)
                
    #         else:
    #             clusterFinal.append(clusterArr)
    #             clusterArr = [p2]

    #   if clusterFinal:
    #         pointsDis = self.distance(clusterFinal[0][0], clusterArr[-1])
    #         if pointsDis <= cluster_threshold:
    #           clusterFinal[0] = clusterArr + clusterFinal[0]
    #         else:
    #           clusterFinal.append(clusterArr)
    #   else:
    #     clusterFinal.append(clusterArr)
      
    # return clusterFinal
    
    
 



if __name__ == "__main__":
    main()

