import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa

dataset = np.genfromtxt("hw08_data_set.csv", delimiter=",", skip_header=True)

N = dataset.shape[0]
K = 5
delta = 1.25
R = 5

class_means = np.array([[+2.5, +2.5],
                        [-2.5, +2.5],
                        [-2.5, -2.5],
                        [+2.5, -2.5],
                        [0, 0]])
class_deviaions = np.array([[[+0.8, -0.6], [-0.6, +0.8]],
                             [[+0.8, +0.6], [+0.6, +0.8]],
                             [[+0.8, -0.6], [-0.6, +0.8]],
                             [[+0.8, +0.6], [+0.6, +0.8]],
                             [[+1.6, 0], [0, +1.6]]])
class_sizes = np.array([50, 50, 50, 50, 100]) 

plt.figure(figsize=(6, 6))
plt.plot(dataset[:, 0], dataset[:, 1], '.', markersize=10, color="black")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

distance_matrix = np.zeros((N, N))
B = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        distance_matrix[i][j] = np.sqrt((dataset[j][0] - dataset[i][0]) ** 2 + (dataset[j][1] - dataset[i][1]) ** 2)
        if distance_matrix[i][j] < delta:
            B[i][j] = 1
        else:
            B[i][j] = 0


plt.figure(figsize=(6, 6))
for i in range(N):
    for j in range(N):
        if B[i][j] == 1:
            plt.plot([dataset[i][0], dataset[j][0]], [dataset[i][1], dataset[j][1]], "gray", linewidth=0.5)
plt.plot(dataset[:, 0], dataset[:, 1], '.', markersize=10, color="black")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

D = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if B[i][j] == 1:
            D[i][i] += 1

L_sym = np.identity(N) - np.matmul(np.matmul(np.sqrt(np.linalg.inv(D)),B),np.sqrt(np.linalg.inv(D)))

eigenvalues, eigenvectors = np.linalg.eig(L_sym)
eigenvectors = eigenvectors[:, eigenvalues.argsort()][:R]
Z = np.transpose(eigenvectors)
centroids =  np.vstack([Z[39], Z[143], Z[204], Z[271], Z[277]])

def update_centroids(memberships, X):
    if memberships is None:
        # initialize centroids
        centroids = X[np.random.choice(range(N), K, False),]
    else:
        # update centroids
        centroids = np.vstack([np.mean(X[memberships == k,:], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)

def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#ff7f00","#33a02c","#e31a1c","#1f78b4","#6a3d9a" ])
    if memberships is None:
        plt.plot(X[:,0], X[:,1], ".", markersize = 10, color = "black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize = 12, 
                 markerfacecolor = cluster_colors[c], markeredgecolor = "black")
    plt.xlabel("x1")
    plt.ylabel("x2")
memberships = None
iteration = 1
while True:
    print("Iteration#{}:".format(iteration))
    old_centroids = centroids
    centroids = update_centroids(memberships, dataset)
    if np.alltrue(centroids == old_centroids):
        break
    old_memberships = memberships
    memberships = update_memberships(centroids, dataset)
    if np.alltrue(memberships == old_memberships):
        plt.show()
        break
    iteration = iteration + 1

plt.figure(figsize = (6, 6))    
plot_current_state(centroids, memberships, dataset)
plt.show()