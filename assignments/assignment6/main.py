import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dt

data_set = np.genfromtxt("hw06_images.csv", delimiter=",")
labels = (np.genfromtxt("hw06_labels.csv")).astype(int)
training_images = data_set[:1000, :]
test_images = data_set[1000:, :]
training_labels = labels[:1000]
test_labels = labels[1000:]

N_train = len(training_labels)
N_test = len(test_labels)

s = 10
C = 10
epsilon = 1e-3


def gaussian_kernel(x1, x2, s):
    D = dt.cdist(x1, x2)
    K = np.exp(-D ** 2 / (2 * s ** 2))
    return (K)


def vector_classification(X_train, y_train, N_train, C, s):
    # calculate Gaussian kernel
    K_train = gaussian_kernel(X_train, X_train, s)
    yyK = np.matmul(y_train[:, None], y_train[None, :]) * K_train

    # set learning parameters
    epsilon = 1e-3

    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N_train, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
    h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
    A = cvx.matrix(1.0 * y_train[None, :])
    b = cvx.matrix(0.0)

    # use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N_train)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C

    # find bias parameter
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(
        y_train[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))

    return alpha, w0


def calculate_predictions(y_predicted, N_train):
    predictions = np.zeros(N_train)
    for i in range(len(y_predicted[0])):
        predictions[i] = (np.argmax(np.array([y_predicted[c][i] for c in range(5)])) + 1).astype(int)
    return predictions


def one_versus_all_training(training_images, training_labels, N_train, c, s):
    y_predicted = np.zeros((5, N_train))
    for i in range(5):
        labels = np.array([1 if training_labels[c] == i + 1 else -1 for c in range(len(training_labels))])
        alpha, w0 = vector_classification(training_images, labels, N_train, c, s)
        K_train = gaussian_kernel(training_images, training_images, s)
        f_predicted = np.matmul(K_train, labels[:, None] * alpha[:, None]) + w0
        y_predicted[i] = np.reshape(f_predicted, N_train)
    return calculate_predictions(y_predicted, N_train).astype(int)


def one_versus_all_test(training_images, training_labels, N_train, N_test, c, s):
    y_predicted = np.zeros((5, N_test))
    for i in range(5):
        labels = np.array([1 if training_labels[c] == i + 1 else -1 for c in range(len(training_labels))])
        alpha, w0 = vector_classification(training_images, labels, N_train, c, s)
        K_train = gaussian_kernel(test_images, training_images, s)
        f_predicted = np.matmul(K_train, labels[:, None] * alpha[:, None]) + w0
        y_predicted[i] = np.reshape(f_predicted, N_test)
    return calculate_predictions(y_predicted, N_test).astype(int)


# Training
y_predicted = one_versus_all_training(training_images, training_labels, N_train, C, s)
confusion_matrix = pd.crosstab(np.reshape(y_predicted, N_train), training_labels, rownames=['y_predicted'],
                               colnames=['y_train'])
print(confusion_matrix)

# Test
y_predicted = one_versus_all_test(training_images, training_labels, N_train, N_test, C, s)
confusion_matrix = pd.crosstab(np.reshape(y_predicted, N_test), test_labels, rownames=['y_predicted'],
                               colnames=['y_test'])
print(confusion_matrix)

c_values = [0.1, 1, 10, 100, 1000]
training_acc = []
_acc = []
for c in c_values:
    y_predicted = one_versus_all_training(training_images, training_labels, N_train, c, s)
    _acc = np.array([1 if y_predicted[i] == training_labels[i] else 0 for i in range(len(y_predicted))])
    training_acc.append(np.sum(_acc) / len(y_predicted))

print('Training accuracies:')
print(training_acc)

test_acc = []
for c in c_values:
    y_predicted = one_versus_all_test(training_images, training_labels, N_train, N_test, c, s)
    _acc = np.array([1 if y_predicted[i] == test_labels[i] else 0 for i in range(len(y_predicted))])
    test_acc.append(np.sum(_acc) / len(y_predicted))
print('Test accuracies:')
print(test_acc)

plt.figure(figsize=(10, 5))
plt.scatter(range(len(c_values)), training_acc, color="blue", label="training")
plt.plot(range(len(c_values)), training_acc, color="blue")
plt.scatter(range(len(c_values)), test_acc, color="red", label="test")
plt.plot(range(len(c_values)), test_acc, color="red")

c_values = ['10^-1', '10^0', '10^1', '10^2', '10^3']
plt.xlabel("Regularization Parameter (C)")
plt.ylabel("Accuracy")
plt.xticks(range(len(c_values)), c_values)
plt.legend(loc='upper left')
plt.show()
