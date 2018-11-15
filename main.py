import os
import random
import numpy as np


N = 200 #Number of data points
M = 10 #Number of features for each data point
#M = 100 #Number of features for each data point LARGE data set only
file = 'data/CS170_SMALLtestdata__96.txt' #path to data set txt
#file = 'data/CS170_LARGEtestdata__58.txt'

def readdataset():
    # Read in data
    data = []
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(fileDir, file)
    with open(filename, "r") as infile:
        for line in infile:
            row_list = []
            data_list = line.split()
            for i in range(M+1):
                row_list.append(float(data_list[i]))
            data.append(row_list)

    for i in range(N):
        print(i+1, data[i])

    return data

def crossvalidation(data, currentfeatures, j):
    return random.randint(1, 101)

def main():

    data = readdataset()
    print()
    print()
    print("Forward selection")

    # Forward selection
    currentfeatures = []
    for i in range(1,M+1):
        print("On the ", i, "th level of the search tree", sep='')
        addfeature = []
        bestaccuracy = 0
        for j in range(1,M+1):
            if j not in currentfeatures:
                print("Considering adding", j, "feature")
                accuracy = crossvalidation(data, currentfeatures, j)
                if accuracy > bestaccuracy:
                    bestaccuracy = accuracy
                    addfeature = j
        currentfeatures.append(addfeature)
        print('Level', i, 'added feature', addfeature, 'to current set')
    print()
    print(currentfeatures)

    print()
    print("Backward selection")

    # Backward selection
    currentfeatures = []
    for i in range(1, M+1):
        currentfeatures.append(i)

    for i in range(1, M + 1):
        print("On the ", i, "th level of the search tree", sep='')
        removefeature = []
        bestaccuracy = 0
        for j in range(1, M + 1):
            if j in currentfeatures:
                print("Considering removing", j, "feature")
                accuracy = crossvalidation(data, currentfeatures, j)
                if accuracy > bestaccuracy:
                    bestaccuracy = accuracy
                    removefeature = j
        currentfeatures.remove(removefeature)
        print('Level', i, 'removed feature', removefeature, 'to current set')
    print()
    print(currentfeatures)





main()