'''
Created on 05/05/2017
@author: Madhura Dole
'''

from sklearn import svm


class SVMClassifier():
    def __init__(self, finalList, classList, testFinalList, testClassList):
        self.finalList = finalList
        self.classList = classList
        self.testFinalList = testFinalList
        self.testClassList = testClassList

    def classify(self):
        svm1 = svm.LinearSVC(C=10, loss='squared_hinge', penalty='l2', tol=0.00001)
        svm1.fit(self.finalList, self.classList)
        scoreSVM = svm1.score(self.testFinalList, self.testClassList)
        print("Support Vector Machines: ", scoreSVM)
