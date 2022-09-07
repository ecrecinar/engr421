import math
import numpy as np
import pandas as pd

images = np.genfromtxt("hw02_images.csv", delimiter=",")
labels = np.genfromtxt("hw02_labels.csv", delimiter=",").astype(int)

K = np.max(labels)
N = images.shape[0]

training_images = []
training_labels = []

test_images = []
test_labels = []

for i in range(30000):
    training_images.append(images[i])
    training_labels.append(labels[i])

for i in range(5000):
    test_images.append(images[i + 30000])
    test_labels.append(labels[i + 30000])

training_images = np.array(training_images)
training_labels = np.array(training_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

training_class_sizes = [0 for i in range(5)]
for i in range(len(training_labels)):
    val = training_labels[i]
    training_class_sizes[val - 1] = training_class_sizes[val - 1] + 1

sample_means = [np.mean(training_images[training_labels == (c + 1)], axis=0) for c in range(K)]
# print("Sample means:")
# print(sample_means)

class_priors = [np.mean(training_labels == (c + 1)) for c in range(K)]
# print("Class priors:")
# print(class_priors)

# sample covariances
sample_deviations = [np.sqrt(np.mean((training_images[training_labels == (c + 1)] - sample_means[c]) ** 2, axis=0)) for
                     c in range(K)]
print("Sample deviations: ")
print(sample_deviations)


def calculate_score(x):
    score = np.stack([-0.5 * np.log(2 * math.pi * sample_deviations[c] ** 2)
                      - 0.5 * (x - sample_means[c])**2 / sample_deviations[c]**2 + np.log(class_priors[c]) for c
                      in
                      range(K)])
    return score


training_score = calculate_score(training_images)
train_predictions = np.argmax(np.mean(training_score[0], axis=1)) + 1
print(train_predictions)
training_confusion_matrix = pd.crosstab(train_predictions, training_labels, rownames=['y_pred'], colnames=['y_truth'])
print("Training confusion matrix:")
print(training_confusion_matrix)

test_score = calculate_score(test_images)
test_pred = np.argmax(test_score, axis=1) + 1
test_confusion_matrix = pd.crosstab(test_score, test_labels, rownames=['y_pred'], colnames=['y_truth'])
print("Test confusion matrix:")
print(test_confusion_matrix)
