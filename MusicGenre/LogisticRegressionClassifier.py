'''
Created on 05/05/2017
@author: Madhura Dole
'''

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


class LogisticRegressionClassifier:
    def __init__(self, finalList, classList, testFinalList, testClassList):
        self.finalList = finalList
        self.classList = classList
        self.testFinalList = testFinalList
        self.testClassList = testClassList

    def classify(self):
        acc = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                                 class_weight=None, random_state=None, solver='liblinear', max_iter=500,
                                 multi_class='ovr',
                                 verbose=0, warm_start=False, n_jobs=1)
        acc.fit(self.finalList, self.classList)
        scoreLR = acc.score(self.testFinalList, self.testClassList)
        scoreLRPredict = acc.predict(self.testFinalList)
        # recall_LR = recall_score(testClassList, scoreLRPredict)
        #  precision_LR = precision_score(testClassList, scoreLRPredict)
        confusion_matrix_LR = confusion_matrix(self.testClassList, scoreLRPredict)
        print("Logistic Regression score: ", scoreLR)
        # print("Logistic Regression recall: ", recall_LR)
        # print("Logistic Regression: ", precision_LR)
        # print("Logistic Regression: ", scoreLR)
        print("Logistic Regression: ", confusion_matrix_LR)
