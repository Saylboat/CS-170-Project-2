import os
import numpy as np


N = 200 #Number of data points
#M = 10 #Number of features for each data point
M = 100 #Number of features for each data point LARGE data set only

#path to data set txt
#file = 'data/CS170_SMALLtestdata__96.txt'
#file = 'data/CS170_SMALLtestdata__108.txt'
#file = 'data/CS170_SMALLtestdata__109.txt'
#file = 'data/CS170_SMALLtestdata__110.txt'
file = 'data/CS170_LARGEtestdata__58.txt'
#file = 'data/CS170_SMALLtestdata__SAMPLE.txt'

def readdataset():
    # Read in data
    datapy = []
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(fileDir, file)

    with open(filename, "r") as infile: #read in data from file
        for line in infile:
            row_list = []
            data_list = line.split()
            for i in range(M+1):
                row_list.append(float(data_list[i]))
            datapy.append(row_list)

    datanp = np.array(datapy) #turn in numpy array for easier functionality

    return datanp

def createtestsets(data, currentfeatures, entry):
    # create training data set and leave on out test point based on current features
    # data is dataset currently being worked on
    # currentfeatures to be tested
    # entry is the row that the testdata point will be

    testdatatmp = []
    testdatatmp.append(data[entry][0])
    for j in currentfeatures:
        testdatatmp.append(data[entry][j])
    testdata = np.array(testdatatmp)

    #print(testdata)

    traindatatmp = []
    for k in range(len(data)):
        traininit = []
        traininit.append(data[k][0])
        for l in currentfeatures:
            traininit.append(data[k][l])
        traindatatmp.append(traininit)
    traindata = np.array(traindatatmp)
    traindata = np.delete(traindata, entry, 0)

    #print(traindata)

    return traindata, testdata

def nearestneighbor(traindata, testdata):
    # nearest neighbors

    distances = []
    for i in range(len(traindata)):
        distance = np.sqrt(np.sum(np.square(testdata[1:] - traindata[i, 1:])))#find euclidean distance
        distances.append([distance, i])
    distances = sorted(distances)
    return distances[0]#return index of point with shortest distance

def leave_one_out_crossvalidation(data, currentfeatures, j, choice):
    # Leave one out cross validation
    # testdata is the data point thats left out
    # traindata is the data set with testdata removed

    features = []
    for i in currentfeatures:
        features.append(i)
    if choice == 1:
        features.append(j)
    elif choice == 2:
        features.remove(j)

    closest = []
    testclasses = []
    trainclasses = []
    numcorrect = 0
    for i in range(len(data)):
        traindata, testdata = createtestsets(data, features, i)
        closest = nearestneighbor(traindata, testdata)
        testclasses.append(testdata[0])  # save class of test case
        trainclasses.append(traindata[closest[1]][0])  # save class of nearest neighbor to test cas
    for i in range(len(trainclasses)):
        if testclasses[i] == trainclasses[i]:  # Compare classes of test cases and classes of nearest neighbors
            numcorrect = numcorrect + 1
    accuracy = (float(numcorrect) / len(data)) * 100
    return accuracy

def backwardsselection(data, choice):
    # Backward selection for features

    print()
    print("Backward selection")
    currentfeatures = []
    for i in range(1, M + 1):
        currentfeatures.append(i)
    for i in range(1, M + 1):
        print("On the ", i, "th level of the search tree", sep='')
        feature = []
        bestaccuracy = 0
        for j in range(1, M + 1):
            if j in currentfeatures:
                accuracy = leave_one_out_crossvalidation(data, currentfeatures, j, choice)
                print("Considering removing feature", j, "with accuracy", accuracy)
                if accuracy > bestaccuracy:
                    bestaccuracy = accuracy
                    feature = j
        currentfeatures.remove(feature)
        print('Level ', i, ': removed feature ', feature, ' from current set with accuracy ', bestaccuracy, '%', sep='')
        print("Feature list: ", currentfeatures)
        print()

def forwardselection(data, choice):
    # Forward selection for features

    print()
    print("Forward selection")
    currentfeatures = []
    for i in range(1, M + 1):
        print("On the ", i, "th level of the search tree", sep='')
        feature = []
        bestaccuracy = 0
        for j in range(1, M + 1):
            if j not in currentfeatures:
                accuracy = leave_one_out_crossvalidation(data, currentfeatures, j, choice)
                print("Considering adding feature", j, "with accuracy", accuracy)
                if accuracy > bestaccuracy:
                    bestaccuracy = accuracy
                    feature = j
        currentfeatures.append(feature)
        print('Level ', i, ': added feature ', feature, ' to current set with accuracy ', bestaccuracy, '%', sep='')
        print("Feature list: ", currentfeatures)
        print()

def holdit(data, feature):
    closest = []
    testclasses = []
    trainclasses = []
    numcorrect = 0
    for i in range(len(data)):
        traindata, testdata = createtestsets(data, [6, 5, 4], i)
        closest = nearestneighbor(traindata, testdata)
        testclasses.append(testdata[0])  # save class of test case
        trainclasses.append(traindata[closest[1]][0])  # save class of nearest neighbor to test cas
    for i in range(len(trainclasses)):
        if testclasses[i] == trainclasses[i]:  # Compare classes of test cases and classes of nearest neighbors
            numcorrect = numcorrect + 1
    accuracy = (float(numcorrect) / len(data)) * 100
    print(traindata[89])
    print(testdata[1:])
    print(accuracy)

def main():

    data = readdataset()

    print("Welcome to Derek Sayler's nearest neighbor classifier with feature selection program")
    print("1. Forward Selection")
    print("2. Backwards Selection")
    print("3. Derek's Custom Algorithm")
    print()
    print("Enter which algorithm you want")
    choice = int(input())
    if choice is 1:
        forwardselection(data, choice)
    elif choice is 2:
        backwardsselection(data, choice)
    else:
        print('ERROR: Invalid option')
    return

main()