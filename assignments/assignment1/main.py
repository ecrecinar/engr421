import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd

np.random.seed(300)

# given data
class_means = np.array([[+0.0, +2.5], [-2.5, -2.0], [+2.5, -2.0]])
class_deviations = np.array([
    [
        [+3.2, +0.0], [+0.0, +1.2]],
    [
        [+1.2, +0.8], [+0.8, +1.2]
    ],
    [
        [+1.2, -0.8], [-0.8, +1.2]
    ]])
class_sizes = np.array([120, 80, 100])

# class samples
points1 = np.random.multivariate_normal(class_means[0, :], class_deviations[0, :, :], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1, :], class_deviations[1, :, :], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2, :], class_deviations[2, :, :], class_sizes[2])

points = np.concatenate((points1, points2, points3))
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2])))
K = np.max(y)
N = sum(class_sizes)
print(points)
plt.figure(figsize=(6, 6))
plt.plot(points1[:, 0], points1[:, 1], "r.", markersize=10)
plt.plot(points2[:, 0], points2[:, 1], "g.", markersize=10)
plt.plot(points3[:, 0], points3[:, 1], "b.", markersize=10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# sample means
sample_means = [np.mean(points[y == (c + 1)], axis=0) for c in range(K)]
print("Sample means:")
print(sample_means)

# sample covariances
sample_covariances = [
    (np.matmul(np.transpose(points[y == (c + 1)] - sample_means[c]), (points[y == (c + 1)] - sample_means[c])) /
     class_sizes[c]) for c
    in range(K)]
print("Sample covariances:")
print(sample_covariances)

# class priors
class_priors = [np.mean(y == (c + 1)) for c in range(K)]
print("Class priors:")
print(class_priors)

# question 4

data_interval = np.linspace(-7, +7, 1401)

Wc = np.array([-0.5 * np.linalg.inv(sample_covariances[c]) for c in range(K)])
wc = np.array([np.matmul(sample_means[c], np.linalg.inv(sample_covariances[c])) for c in range(K)])
wc0 = np.array([-0.5 * (np.matmul(np.matmul(np.transpose(sample_means[c]), np.linalg.inv(sample_covariances[c])),sample_means[c]))
                - 0.5 * np.log(np.linalg.det(sample_covariances[c]))
                + np.log(class_priors[c]) for c in range(K)])

score_values = np.stack([np.matmul(np.matmul(np.transpose(points[i]), Wc[c]), points[i])
                         + np.matmul(np.transpose(wc[c]), points[i])
                         + wc0[c]
                         for c in range(K)]
                        for i in range(len(points))
)
print(score_values)
pred = np.argmax(score_values, axis=1) + 1
print(pred)

confusion_matrix = pd.crosstab(pred, y, rownames=['y_pred'], colnames=['y_truth'])
print("Confusion matrix:")
print(confusion_matrix)


# question 5

x1_interval = np.linspace(-6, +6, 1201)
x2_interval = np.linspace(-6, +6, 1201)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))

for c in range(K):
    discriminant_values[:, :, c] = (Wc[c, 0, 0] * x1_grid ** 2) + (Wc[c, 0, 1] * x1_grid * x2_grid) + (
                wc[1, 0] * x2_grid * x1_grid) + (wc[1, 1] * x2_grid ** 2) + (wc[c, 0] * x1_grid) + (
                                               wc[c, 1] * x2_grid) + wc0[c]

A = discriminant_values[:, :, 0]
B = discriminant_values[:, :, 1]
C = discriminant_values[:, :, 2]
A[(A < B) & (A < C)] = np.nan
B[(B < A) & (B < C)] = np.nan
C[(C < A) & (C < B)] = np.nan
discriminant_values[:, :, 0] = A
discriminant_values[:, :, 1] = B
discriminant_values[:, :, 2] = C

plt.figure(figsize=(8, 8))
plt.plot(points[y == 1, 0], points[y == 1, 1], "r.", markersize=10)
plt.plot(points[y == 2, 0], points[y == 2, 1], "g.", markersize=10)
plt.plot(points[y == 3, 0], points[y == 3, 1], "b.", markersize=10)

plt.plot(points[pred != y, 0], points[pred != y, 1], "o", markersize=12, fillstyle="none")

plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 1], levels=0, colors="k")
plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 2], levels=0, colors="k")
plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 1] - discriminant_values[:, :, 2], levels=0, colors="k")

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()