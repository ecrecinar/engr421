import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

np.random.seed(421)

class_means = np.array([-3.0,-1.0, +3.0])
class_deviations = np.array([1.2,1.0,1.3])
class_sizes = np.array([40,30,50])

points1 = np.random.normal(class_means[0], class_deviations[0],class_sizes[0])
points2 = np.random.normal(class_means[1], class_deviations[1],class_sizes[1])
points3 = np.random.normal(class_means[2], class_deviations[2],class_sizes[2])

points = np.concatenate((points1,points2,points3))

y = np.concatenate((np.repeat(1,class_sizes[0]),np.repeat(2,class_sizes[1]),np.repeat(3,class_sizes[2])))

print(points)
print(y)


# np.savetxt("lab01_dataset.csv", np.stack((points,y),axis=1), delimiter=",")

data_interval = np.linspace(-7, +7, 1401)
print(data_interval)

density1 = stats.norm.pdf(data_interval, loc = class_means[0], scale = class_deviations[0])
density2 = stats.norm.pdf(data_interval, loc = class_means[1], scale = class_deviations[1])
density3 = stats.norm.pdf(data_interval, loc = class_means[2], scale = class_deviations[2])

plt.figure(figsize=(10,6))
plt.plot(data_interval, density1, "r")
plt.plot(data_interval, density2, "g")
plt.plot(data_interval, density3, "b")
plt.plot(points1, np.repeat(-0.01, class_sizes[0]), "r.", markersize = 12)
plt.plot(points2, np.repeat(-0.02, class_sizes[1]), "g.", markersize = 12)
plt.plot(points3, np.repeat(-0.03, class_sizes[2]), "b.", markersize = 12)
plt.xlabel("x")
plt.ylabel("density")
plt.show()

## importing data

dataset = np.genfromtxt("lab01_dataset.csv", delimiter=",")
x = dataset[:,0]
y = dataset[:,1].astype(int)

K = np.max(y)
N = dataset.shape[0]

print(K,N)
print(range(K))
print(np.mean(x[y == (0+1)]))
print(np.mean(x[y == (1+1)]))
print(np.mean(x[y == (2+1)]))

sample_means = [np.mean(x[y == (c+1)]) for c in range(K)]
print(sample_means)

sample_deviations = [np.sqrt(np.mean((x[y == (c+1)] - sample_means[c])**2)) for c in range(K)]
print(sample_deviations)

class_priors = [np.mean(y == (c+1)) for c in range(K)]
print(class_priors)
print(np.sum(class_priors))

data_interval = np.linspace(-7,+7,1401)
print(data_interval)

score_values = np.stack([-0.5 * np.log(2 * math.pi * sample_deviations[c]**2)
                         -0.5 * (data_interval - sample_means[c])**2 / sample_deviations[c]**2
                         + np.log(class_priors[c])
    for c in range(K)
])

print(score_values)
print(score_values.shape)

plt.figure(figsize= (10,6))
plt.plot(data_interval, score_values[0,:], "r")
plt.plot(data_interval, score_values[1,:], "g")
plt.plot(data_interval, score_values[2,:], "b")

plt.xlabel("x")
plt.ylabel("score")
plt.show()

np.log(np.exp(5000) + np.exp(6000) + np.exp(7000))

