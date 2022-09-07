import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def safelog2(x):
    if x == 0:
        return (0)
    else:
        return (np.log2(x))


# read data into memory
dataset = np.genfromtxt("hw05_data_set.csv", delimiter=",")

training_data = []
test_data = []
for i in range(150):
    training_data.append(dataset[i])
for i in range(122):
    test_data.append(dataset[i + 150])

training_data = np.array(training_data)
test_data = np.array(test_data)

train_indices = np.arange(0, 125)
test_indices = np.arange(126, 277)

x_train = training_data[:, 0]
x_test = test_data[:, 0]
y_train = training_data[:, 1].astype(int)
y_test = test_data[:, 1].astype(int)
D = training_data.shape[1]
N_train = len(y_train)
N_test = len(y_test)

# create necessary data structures
node_indices = {}
is_terminal = {}
need_split = {}
node_splits = {}
node_means = {}

# put all training instances into the root node
node_indices[1] = np.array(range(N_train))
is_terminal[1] = False
need_split[1] = True


def decision_tree(P):
    while True:
        # find nodes that need splitting
        split_nodes = [key for key, value in need_split.items() if value == True]
        # check whether we reach all terminal nodes
        if len(split_nodes) == 0:
            break
        # find best split positions for all nodes
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False

            if len(data_indices) <= P :
                is_terminal[split_node] = True
            else:
                is_terminal[split_node] = False

                mean = np.mean(y_train[data_indices])
                best_scores = np.repeat(0.0, D)
                best_splits = np.repeat(0.0, D)
                for d in range(D):
                    unique_values = np.sort(np.unique(x_train[data_indices, d]))
                    split_positions = (unique_values[1:len(unique_values)] + unique_values[
                                                                             0:(len(unique_values) - 1)]) / 2
                    split_scores = np.repeat(0.0, len(split_positions))
                    for s in range(len(split_positions)):
                        error = 0
                        left_indices = data_indices[x_train[data_indices, d] > split_positions[s]]
                        right_indices = data_indices[x_train[data_indices, d] <= split_positions[s]]
                        left_mean = np.mean(y_train[left_indices])
                        right_mean = np.mean(y_train[right_indices])
                        error += np.sum((y_train[right_indices] - right_mean) ** 2)
                        error += np.sum((y_train[left_indices] - left_mean) ** 2)
                        split_scores[s] = error / len(right_indices) + len(left_indices)
                    best_scores[d] = np.min(split_scores)
                    best_splits[d] = split_positions[np.argmin(split_scores)]
                # decide where to split on which feature
                split_d = np.argmin(best_scores)

                node_splits[split_node] = best_splits[split_d]

                # create left node using the selected split
                left_indices = data_indices[x_train[data_indices, split_d] > best_splits[split_d]]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True

                # create right node using the selected split
                right_indices = data_indices[x_train[data_indices, split_d] <= best_splits[split_d]]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True






plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, "b.", markersize=10)
plt.plot(x_test, y_test, "r.", markersize=10)

plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training", "test"])
plt.show()
