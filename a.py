# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 22:53:50 2017
@author: Abhijeet Singh
"""
import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix


def make_Dictionary(train_dir):
    all_words = []
    with open(train_dir) as m:
        for line in m:
            words = line.split()
            words = words[1:]
            all_words += words
    dictionary = Counter(all_words)

    for item in list(dictionary): # this works with python 3.x version
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary


def extract_features(mail_dir):
    nb = 0
    lineID = 0
    with open(mail_dir) as fi:
        for n in fi:
            nb += 1

    print(nb)
    features_matrix = np.zeros((int(nb), 3000))
    labels_matrix = np.zeros((int(nb), 1))

    with open(mail_dir) as fi:
        for line in fi:
            words = line.split()

            #labels matrix
            if words[0] == 'ham':
                labels_matrix[lineID, 0] = 0
            elif words[0] == 'spam':
                labels_matrix[lineID, 0] = 1

            words = words[1:]

            #features matrix
            for word in words:
                wordID = 0
                for i, d in enumerate(dictionary):
                    if d[0] == word:
                        wordID = i
                        features_matrix[lineID, wordID] = words.count(word)

            lineID = lineID + 1

    return labels_matrix, features_matrix


# Create a dictionary of words with its frequency
train_dir = 'messages.txt'
dictionary = make_Dictionary(train_dir)
print(dictionary)

train_labels, train_matrix = extract_features(train_dir)
for l in train_labels:
    print(l)

# Training SVM and Naive bayes classifier and its variants
model1 = LinearSVC()
model2 = MultinomialNB()
model1.fit(train_matrix, train_labels)
model2.fit(train_matrix, train_labels)


# Test the unseen mails for Spam
test_dir = 'messages_test.txt'
test_labels, test_matrix = extract_features(test_dir)

result1 = model1.predict(test_matrix)
result2 = model2.predict(test_matrix)

print(confusion_matrix(test_labels, result1))
print(confusion_matrix(test_labels, result2))
