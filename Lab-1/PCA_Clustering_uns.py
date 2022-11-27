import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io
from sklearn.manifold import TSNE
import itertools
import matplotlib.cm as cm


def eigens_retrieval(Z):
    #covariance matrix
    cov = np.cov(Z.T) 
    #Eigen values and Eigen vectors
    val,vec = np.linalg.eig(cov) 
    return val,vec

# Helper function for picking the first element for sorting
def first(n):  
    return n[0] 

# Perform a principal component analysis
def PCA(X,alpha=0 , d=0):
    
    # Calculate the mean of every feature by calculating the mean 
    # Over the X-axis
    means = np.mean(X, axis=0)
    
    # Center the data by subtracting the features-mean from 
    # Every feature in the data
    Z = np.copy(X)
    for i, x in enumerate(means):
        # print(x)
        Z[::, i] = X[::, i]-x
        
    # Calculate eigenvalues and eigenvectors
    val, vec = eigens_retrieval(Z)

    # Combine the Eigenvalues and Eigenvectors so that when the
    # Eigenvalues get sorted, the Eigenvectors are taken with.
    zipped = []
    np.asarray(zipped)
    for i in range(len(val)):
        zipped.append((val[i], list(vec[i])))
    zipped = sorted(zipped, key = first)
    zipped = zipped[::-1]
    # set the eigenvalues and eigenvectors back into their own variables
    # to make computation easier for the next steps
    val = []
    vec = []
    for i in zipped:
        val.append(i[0])
        vec.append(i[1])
    val = np.array(val)
    vec = np.array(vec)
    
    # If no alpha value was given but the dimensionality then we
    # Assume a dimensionality was given and we use that
    # Otherwise we calculate the dimensionality for the threshold alpha
    if alpha == 0:
        # return 3 variables, the principal component vectors
        # the eigenvalues and the reduced centered data
        return vec[:d],val[:d],Z.dot(vec[:d].T)
    
    p = 1 #total variance starting at 100%
    d = len(val) #dimensionality which starts at n (original dimension)
    # While we have not passed the threshold decrease the dimensionality,
    # and recalculate the variance fraction p
    while alpha <= p:
        d-=1
        p = np.sum(val[:d])/np.sum(val)
    # Increase dimension by one again since that is the correct dimension
    d+=1
    p = np.sum(val[:d])/np.sum(val)

        # return 3 variables, the principal component vectors
        # the eigenvalues and the reduced centered data
    return vec[:d],val[:d],Z.dot(vec[:d].T)

# function used for plotting the eigenvalues so we now what the values are
# for the nth principle component
def plotting(eig):
    fig = plt.figure()
    frame = fig.add_subplot(1,1,1)
    eig = eig[:20]
    x = np.arange(len(eig))[:20]+1
    frame.plot(x,eig,"bx")
    frame.set_xlabel("nth eigenvalue")
    frame.set_ylabel("value of the eigenvalue")
    frame.set_title("plotting the eigenvalues in decreasing order")

    plt.show()

# function for plotting the data points in 2 dimensions using TSNE to get
# image data to 2 dimensions.
def plot_tsne(data):
    x_emb = TSNE().fit_transform(data)

    fig = plt.figure()
    frame = fig.add_subplot(1,1,1)
    labels = list(itertools.repeat(0,72))
    for i in range(1,20):
        labels = labels + list(itertools.repeat(i,72))
    frame.scatter(x_emb[::,0],x_emb[::,1],c=labels,s=4)
    """ Does not work """
    # colors = cm.rainbow(np.linspace(0,1,20))
    # for i in range(1,21):
    #     print(i)
    #     frame.scatter(x_emb[:i*72:,0],x_emb[:72*i:,1],color = colors[i-1],s=4)
    frame.set_xlabel("dimension 1")
    frame.set_ylabel("dimension 2")
    frame.set_title("Data values in two dimensions after using TSNE")
    frame.legend()
    plt.show()

    
    
def main():
    # Load the file as a dictionary
    mat = scipy.io.loadmat("/Users/newjessar/Documents/GitHub/ML-RUG/Lab-1/COIL20.mat")

    # Extract the data from the dictionary
    X = np.array(mat['X']) 

    ## Calculate the Covariance matrix using a numpy function,
    ## and then calculate the eigenvalues and eigenvectors from that
    ## Covariance matrix
    # comp,eig,data = PCA(X, alpha = 0.9)
    # plot_tsne(data)
    # # #plotting(eig)

    # #look at the dimension of the matrix for certain variance fractions
    for i in [0.9,0.95,0.98]:
        comp,eig,data = PCA(X,alpha=i)
        print(len(eig))
    
    
    
if __name__ == "__main__":
    main()
