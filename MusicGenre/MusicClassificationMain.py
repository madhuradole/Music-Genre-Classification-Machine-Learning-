'''
Created on 05/05/2017
@author: Madhura Dole
'''

import librosa
import os
import numpy as np

from LogisticRegressionClassifier import LogisticRegressionClassifier
from KNNClassifier import KNNClassifier
from SVMClassifier import SVMClassifier
from PerceptronClassifier import PerceptronClassifier
from DecisionTreeClassifier import DecisionTreeClassifier
from KMeansClassifier import KMeansClassifier

featuresArray = []


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def classfiySVM():
    svmInst = SVMClassifier(finalList, classList, testFinalList, testClassList)
    svmInst.classify()


def classifyKNN():
    knnInst = KNNClassifier(finalList, classList, testFinalList, testClassList)
    knnInst.classify()


def classifyDecisionTree():
    dtInst = DecisionTreeClassifier(finalList, classList, testFinalList, testClassList)
    dtInst.classify()


def classifyKMeans():
    kmInst = KMeansClassifier(finalList, classList, testFinalList, testClassList)
    kmInst.classify()


def classifyPerceptron():
    percInst = PerceptronClassifier(finalList, classList, testFinalList, testClassList)
    percInst.classify()


def classifyLogisticRegression():
    lrInst = LogisticRegressionClassifier(finalList, classList, testFinalList, testClassList)
    lrInst.classify()


if __name__ == '__main__':
    # help menu
    walk_dir = "../genres"
    # walk_dir = "C:\Disk (D)\MSCS UTD\Spring 2017\Adv ML\Project\genres"

    i = 0.0
    print('walk_dir = ' + walk_dir)
    Y = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
         'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
    # counter_list = list(enumerate(Y, 1))
    finalList = []
    classList = []
    testFinalList = []
    testClassList = []
    for root, subdirs, files in os.walk(walk_dir):
        i = 0
        for filename in files:
            i += 1
            if filename.endswith('.au'):
                file_path = os.path.join(root, filename)
                # print('\t- file %s (full path: %s)' % (filename, file_path))
                ppFileName = rreplace(file_path, ".au", ".pp", 1)

                try:
                    signal, fs = librosa.load(file_path)
                    mfccs = librosa.feature.mfcc(signal, sr=fs, n_mfcc=12)
                    l = mfccs.shape[1]
                    mfccs = mfccs[:, int(0.2 * l):int(0.8 * l)]

                    mean = mfccs.mean(axis=1)
                    covar = np.cov(mfccs, rowvar=1)

                    mean.resize(1, mean.shape[0])
                    # it returns matrix.. not useful for machine learning algorithms except KNN
                    npArray = np.concatenate((mean, covar), axis=0)
                    templist = []
                    for ls in np.nditer(npArray):
                        templist.append(ls)
                    if (i > 70):
                        testFinalList.append(templist)
                        testClassList.append(Y[filename.split('.')[0]])
                    else:
                        finalList.append(templist)
                        classList.append(Y[filename.split('.')[0]])


                        # prepossessingAudio(file_path, ppFileName)
                except Exception as e:
                    print ("Error accured" + str(e))

    # Decision Tree
    classifyDecisionTree()

    # KNN
    classifyKNN()

    # Perceptron
    classifyPerceptron()

    # SVM
    classfiySVM()

    # Logistic Regression
    classifyLogisticRegression()

    # KMeans
    # classifyKMeans()
