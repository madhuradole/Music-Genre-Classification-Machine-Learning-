'''
Created on 05/05/2017
@author: Madhura Dole
'''

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import numpy as np


class PerceptronClassifier:
    def __init__(self, finalList, classList, testFinalList, testClassList):
        self.finalList = finalList
        self.classList = classList
        self.testFinalList = testFinalList
        self.testClassList = testClassList

    def classify(self):
        clf_p = Perceptron(penalty=None, alpha=0.01, fit_intercept=True, n_iter=5000, shuffle=True, verbose=0, eta0=1.0,
                           n_jobs=1, random_state=0, class_weight=None, warm_start=False)

        clf_p.fit(self.finalList, self.classList)
        res1 = clf_p.predict(self.testFinalList)
        score_p = clf_p.score(self.testFinalList, self.testClassList)
        print("Predicted Perceptron Output: ", res1)
        print("Pecerptron Score: ", score_p)
