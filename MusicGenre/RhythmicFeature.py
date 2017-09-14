'''
Created on 05/04/2017
@author: Madhura Dole
'''
import librosa
import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score


def main():
    DIR = "genres"


featuresArray = []


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def classifyPerceptron():
    global clf, scoreMLP
    # MLF2
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='sgd', alpha=0.0001, batch_size='auto',
                        learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=200, shuffle=True,
                        random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                        nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                        beta_2=0.999, epsilon=1e-08)
    clf.fit(np.array(finalList), np.array(classList))
    scoreMLP = clf.score(testFinalList, testClassList)
    print("Multilayer Perceptron: ", scoreMLP)

    # Multilayer Perceptron Classification
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                        random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                        nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                        beta_2=0.999, epsilon=1e-08)
    clf.fit(np.array(finalList), np.array(classList))
    scoreMLP = clf.score(testFinalList, testClassList)
    print("Multilayer Perceptron score: ", scoreMLP)
    scoreMLPPredict = clf.predict(testFinalList)
    confusion_matrix_MLP = confusion_matrix(testClassList, scoreMLPPredict)
    print("Multilayer Perceptron CM mat: ", confusion_matrix_MLP)


def classfiySVM():
    svm1 = svm.LinearSVC(C=10, loss='squared_hinge', penalty='l2', tol=0.00001)
    svm1.fit(finalList, classList)
    scoreSVM = svm1.score(testFinalList, testClassList)
    print("Support Vector Machines: ", scoreSVM)


def classifyKMeans():
    kmeans = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto',
                    verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
    kmeans.fit(finalList, classList)
    scoreKMeans = kmeans.score(testFinalList, testClassList)
    print("KMeans: ", scoreKMeans)


def classifyLogisticRegression():
    acc = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                             class_weight=None, random_state=None, solver='liblinear', max_iter=500, multi_class='ovr',
                             verbose=0, warm_start=False, n_jobs=1)
    acc.fit(finalList, classList)
    scoreLR = acc.score(testFinalList, testClassList)
    print("Logistic Regression score: ", scoreLR)

    scoreLRPredict = acc.predict(testFinalList)
    confusion_matrix_LR = confusion_matrix(testClassList, scoreLRPredict)
    print("Logistic Regression: ", confusion_matrix_LR)


if __name__ == '__main__':
    # help menu
    # walk_dir = "C:\Eclipse Workspace\git-repo\genreXpose\genreXpose\genres"
    walk_dir = "../genres"

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
                    signal, sr = librosa.load(file_path)
                    y_harmonic, y_percussive = librosa.effects.hpss(signal)
                    n_mfcc = 5
                    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                                                 sr=sr)
                    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)

                    mfcc_delta = librosa.feature.delta(mfcc)
                    chromagram = librosa.feature.chroma_stft(y=signal,
                                                             sr=sr, n_chroma=12)
                    beat_features = np.vstack([chromagram, mfcc_delta])

                    mean = beat_features.mean(axis=1)
                    covar = np.cov(beat_features, rowvar=1)

                    iu1 = np.triu_indices(mean.shape[0])
                    ravelcovar = covar[iu1]

                    mean.resize(1, mean.shape[0])
                    npArray = np.append(mean, ravelcovar)
                    templist = []
                    for ls in np.nditer(npArray):
                        templist.append(ls)
                    if (i > 3):
                        testFinalList.append(templist)
                        testClassList.append(Y[filename.split('.')[0]])
                    else:
                        finalList.append(templist)
                        classList.append(Y[filename.split('.')[0]])


                        # prepossessingAudio(file_path, ppFileName)
                except Exception as e:
                    print "Error accured" + str(e)

    # MLP
    classifyPerceptron()

    # SVM
    classfiySVM()

    # KMeans
    classifyKMeans()

    # Logistic Regression
    classifyLogisticRegression()
