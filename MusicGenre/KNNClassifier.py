'''
Created on 05/05/2017
@author: Madhura Dole
'''

from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    def __init__(self, finalList, classList, testFinalList, testClassList):
        self.finalList = finalList
        self.classList = classList
        self.testFinalList = testFinalList
        self.testClassList = testClassList

    def classify(self):
        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(self.finalList, self.classList)
        k_score = knn.score(self.testFinalList, self.testClassList)
        print("KNN Score: ", k_score)
