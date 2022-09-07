import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dataset = np.genfromtxt("hw04_data_set.csv", delimiter=",", skip_header=True)
training_data = []
test_data = []
for i in range(150):
    training_data.append(dataset[i])
for i in range(122):
    test_data.append(dataset[i + 150])

training_data = np.array(training_data)
test_data = np.array(test_data)

x_training = training_data[:, 0]
x_test = test_data[:, 0]
y_training = training_data[:, 1].astype(int)
y_test = test_data[:, 1].astype(int)
N_test = x_test.shape[0]

minimum_value = min(x_training)
maximum_value = max(x_training)

bin_width = 0.37
origin = 1.5

# regressogram
left = np.arange(origin, maximum_value, bin_width)
right = np.arange(origin + bin_width, maximum_value + bin_width, bin_width)
data_interval = np.arange(origin, maximum_value, step=0.0001)

p_hat = np.array([np.sum(((left[i] < x_training) & (x_training <= right[i])) * y_training)
                  / np.sum((left[i] < x_training) & (x_training <= right[i]))
                  for i in range(len(left))])

plt.figure(figsize=(10, 6))
plt.plot(x_training, y_training, "b.", markersize=10)
plt.plot(x_test, y_test, "r.", markersize=10)
for i in range(len(left)):
    plt.plot([left[i], right[i]], [p_hat[i], p_hat[i]], "k-")
for i in range(len(left) - 1):
    plt.plot([right[i], right[i]], [p_hat[i], p_hat[i + 1]], "k-")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training", "test"])
plt.show()

# rmse for regressogram
y_hat = np.zeros(N_test)
for j in range(len(left)):
    for i in range(N_test):
        if left[j] < x_test[i] <= right[j]:
            y_hat[i] = p_hat[int((x_test[i] - origin) / bin_width)]

rmse = np.sqrt(np.sum((y_test - y_hat) ** 2 / N_test))
print("Regressogram => RMSE is", rmse, "when h is ", bin_width)

# running mean smoother
p_hat = np.array([np.sum((np.abs((x - x_training) / bin_width) < 1) * y_training)
                  / np.sum(np.abs((x - x_training) / bin_width) < 1) for x in data_interval])

plt.figure(figsize=(10, 6))
plt.plot(x_training, y_training, "b.", markersize=10)
plt.plot(x_test, y_test, "r.", markersize=10)
plt.plot(data_interval, p_hat, "k-")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training", "test"])
plt.show()

# rmse for rms
for i in range(N_test):
    y_hat[i] = p_hat[int((x_test[i] - origin) / 0.0001)]

rmse = np.sqrt(np.sum((y_test - y_hat) ** 2 / N_test))
print("Running Mean Smoother => RMSE is", rmse, "when h is ", bin_width)

# kernel smoother


def K(u):
    return 1 / np.sqrt(math.pi * 2) * np.exp(- u ** 2 / 2)


p_hat = np.array(
    [np.sum(K((x - x_training) / bin_width) * y_training) / np.sum(K((x - x_training) / bin_width)) for x in data_interval])

plt.figure(figsize=(10, 6))
plt.plot(x_training, y_training, "b.", markersize=10)
plt.plot(x_test, y_test, "r.", markersize=10)
plt.plot(data_interval, p_hat, "k-")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training", "test"])
plt.show()

# rmse for kernel smoother
for i in range(N_test):
    y_hat[i] = p_hat[int((x_test[i] - origin) / 0.0001)]

rmse = np.sqrt(np.sum((y_test - y_hat) ** 2 / N_test))
print("Kernel Smoother => RMSE is", rmse, "when h is ", bin_width)