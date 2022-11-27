
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt


# plots dendrograms from the data based on different linkage measures
# do this in either one graph based on one linkage measure or multiple graphs
# in one figure for multiple different linkage measures
def plotDendo(data,linkage,lines,onePlot = False):
    if onePlot:
        fig, frames = plt.subplots(1,1,figsize=(10,10))
        Z = hierarchy.linkage(data, linkage)
        dn = hierarchy.dendrogram(Z,ax=frames)
        for j in lines:
            frames.axhline(j,linestyle="--",color="black")  
        frames.set_title("linkage: " + linkage)
        plt.show()
        return

    fig, frames = plt.subplots(4,1,figsize=(10,8))
    for i,frame in enumerate(frames):
        Z = hierarchy.linkage(data, linkage[i])
        dn = hierarchy.dendrogram(Z,ax=frame)
        for j in lines[linkage[i]]:
            frame.axhline(j,linestyle="--",color="black")  
        frame.set_title("linkage: " + linkage[i])
    fig.tight_layout()
    plt.show()

# this function clusters data into groups for different linkage measures and number of clusters 
# it then plots the data with different labels that are determined from the clustering
def plotClusters(data,linkage,n_clusters,titles,show=True):
    labels = []
    for i in range(len(linkage)):
        labels.append([])
        for j in n_clusters:
            cluster = AgglomerativeClustering(n_clusters=j ,linkage=linkage[i]).fit(data)
            labels[i].append(cluster.labels_)

    fig,frames = plt.subplots(len(linkage),len(n_clusters),figsize=(16,20))
    frames = frames.flat

    for i in range(len(linkage)):
        for j in n_clusters:
            frames[i*len(n_clusters)+(j-n_clusters[0])].scatter(data[::,0],data[::,1],c=labels[i][j-n_clusters[0]])
            frames[i*len(n_clusters)+(j-n_clusters[0])].set_xlabel("x")
            frames[i*len(n_clusters)+(j-n_clusters[0])].set_ylabel("y")
            frames[i*len(n_clusters)+(j-n_clusters[0])].set_title(titles[i*len(n_clusters)+(j-n_clusters[0])])

    fig.tight_layout()
    if show:
        plt.show()
    return labels

# This function computes the average silhouette score for a given clustering method
# the data, labels and number of clusters need to be given
def silDistance(data, labels, n_clusters):
    Sx = 0
    for i,x in enumerate(data):
        clusters = []
        for m in range(n_clusters):
            clusters.append([m,0,0])
        ax = 0
        bx = 0
        dis = 0
        Ci = 0
        for j,y in enumerate(data):
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
    return Sx/len(data)
        
  
def main():

    linkage = ["single","complete","average","ward"]
    data = np.loadtxt("/Users/newjessar/Documents/GitHub/ML-RUG/Lab-2/data_clustering.csv",delimiter = ",")
    lines = {"single": [0.103,0.138,0.148], "complete": [0.48,0.633,0.65], "average":[0.26,0.29,0.31], "ward":[1.05,1.25,2.55]}
    nc = [2,3,4]

    titles = []
    for i in linkage:
        for j in nc:
            title = "linkage: " + i + " / n_clusters: " + str(j)
            titles.append(title)
            
    SxL = []
    labels = plotClusters(data,linkage,nc,titles,show=False)
    print(len(labels))
    for x in labels:
        for j,y in enumerate(x):
            Sx = silDistance(data,y,j+2)
            SxL.append(Sx)
    print(SxL)
    linkage = linkage[3]
    lines = lines[linkage]
    plotDendo(data,linkage,  lines, onePlot=True)
    
if __name__ == "__main__":
    main()
