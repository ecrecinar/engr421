import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def safelog(x):
    return(np.log(x + 1e-100))
np.random.seed(421)
# mean parameters
class_means = np.array([[+1.5, +1.5],
                        [-1.5, -1.5]])
# covariance parameters
class_covariances = np.array([[[+1.6, +1.2],
                               [+1.2, +1.6]],
                              [[+1.6, -1.2],
                               [-1.2, +1.6]]])
# sample sizes
class_sizes = np.array([120, 180])
# generate random samples
points1 = np.random.multivariate_normal(class_means[0,:], class_covariances[0,:,:], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1,:], class_covariances[1,:,:], class_sizes[1])
X = np.vstack((points1, points2))

# generate corresponding labels
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(0, class_sizes[1])))

# write data to a file
np.savetxt("lab03_data_set.csv", np.hstack((X, y[:, None])), fmt = "%f,%f,%d")
# plot data points generated
plt.figure(figsize = (10, 10))
plt.plot(points1[:,0], points1[:,1], "r.", markersize = 10)
plt.plot(points2[:,0], points2[:,1], "b.", markersize = 10)
plt.xlabel("x1")
plt.ylabel("x2")
# plt.show()

# read data into memory
data_set = np.genfromtxt("lab03_data_set.csv", delimiter = ",")

# get X and y values
X = data_set[:,[0, 1]]
y_truth = data_set[:,2].astype(int)

# get number of samples
N = data_set.shape[0]

# define the sigmoid function
def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(X, w) + w0))))

draw_seq = np.linspace(-10, +10, 2001)
plt.figure(figsize = (10, 6))
plt.plot(draw_seq, 1 / (1 + np.exp(-draw_seq)), "r-")
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
# plt.show()

# define the gradient functions
def gradient_w(X, y_truth, y_predicted):
    return(-np.sum(np.transpose(np.repeat([y_truth - y_predicted], X.shape[1], axis = 0)) * X, axis = 0))

def gradient_w0(y_truth, y_predicted):
    return(-np.sum(y_truth - y_predicted))

# set learning parameters
eta = 0.01
epsilon = 1e-3

# randomly initalize w and w0
np.random.seed(421)
w = np.random.uniform(low = -0.01, high = 0.01, size = X.shape[1])
w0 = np.random.uniform(low = -0.01, high = 0.01, size = 1)

print(y_truth)
# learn w and w0 using gradient descent
iteration = 1
objective_values = []
while 1:
    y_predicted = sigmoid(X, w, w0)
    print(y_predicted)
    objective_values = np.append(objective_values, -np.sum(y_truth * safelog(y_predicted) + (1 - y_truth) * safelog(1 - y_predicted)))
    w_old = w
    w0_old = w0

    w = w - eta * gradient_w(X, y_truth, y_predicted)
    w0 = w0 - eta * gradient_w0(y_truth, y_predicted)

    if np.sqrt((w0 - w0_old)**2 + np.sum((w - w_old)**2)) < epsilon:
        break

    iteration = iteration + 1
# print(w, w0)

# plot objective function during iterations
plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
# plt.show()


# calculate confusion matrix
y_predicted = 1 * (y_predicted > 0.5)
confusion_matrix = pd.crosstab(y_predicted, y_truth, rownames = ['y_pred'], colnames = ['y_truth'])
# print(confusion_matrix)

# evaluate discriminant function on a grid
x1_interval = np.linspace(-6, +6, 1201)
x2_interval = np.linspace(-6, +6, 1201)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = w[0] * x1_grid + w[1] * x2_grid + w0

plt.figure(figsize = (10, 10))
plt.plot(X[y_truth == 1, 0], X[y_truth == 1, 1], "r.", markersize = 10)
plt.plot(X[y_truth == 0, 0], X[y_truth == 0, 1], "b.", markersize = 10)
plt.plot(X[y_predicted != y_truth, 0], X[y_predicted != y_truth, 1], "ko", markersize = 12, fillstyle = "none")
plt.contour(x1_grid, x2_grid, discriminant_values, levels = 0, colors = "k")
plt.xlabel("x1")
plt.ylabel("x2")
# plt.show()

