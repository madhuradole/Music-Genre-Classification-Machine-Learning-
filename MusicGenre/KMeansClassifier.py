'''
Created on 05/05/2017
@author: Madhura Dole
'''

from sklearn.cluster import KMeans

class KMeansClassifier:
    def __init__(self, finalList, classList, testFinalList, testClassList):
        self.finalList = finalList
        self.classList = classList
        self.testFinalList = testFinalList
        self.testClassList = testClassList

    def classify(self):
        kmeans = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                        precompute_distances='auto',
                        verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
        kmeans.fit(self.finalList, self.classList)
        scoreKMeans = kmeans.score(self.testFinalList, self.testClassList)
        print("KMeans: ", scoreKMeans)