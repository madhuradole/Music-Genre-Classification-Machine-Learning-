'''
Created on 05/05/2017
@author: Madhura Dole
'''

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


class DecisionTreeClassifier:
    def __init__(self, finalList, classList, testFinalList, testClassList):
        self.finalList = finalList
        self.classList = classList
        self.testFinalList = testFinalList
        self.testClassList = testClassList

    def classify(self):
        regr_1 = DecisionTreeRegressor(max_depth=6)
        regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=300)
        regr_1.fit(self.finalList, self.classList)
        regr_2.fit(self.finalList, self.classList)
        y_1 = regr_1.predict(self.testFinalList)
        y_2 = regr_2.predict(self.testFinalList)
        score1 = regr_1.score(self.testFinalList, self.testClassList)
        score2 = regr_2.score(self.testFinalList, self.testClassList)
        print("Decision tree score1: ", score1, "Adaboost score2: ", score2)
