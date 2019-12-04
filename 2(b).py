from numpy.random import multivariate_normal
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np

def generate_data():
    x = []
    mean = [1, 1]
    cov = [[1,0], [0,1]]
    m, n= multivariate_normal(mean, cov, size=1000).T
    for i in range(len(m)):
        x.append([m[i], n[i]])
    mean = [-1, -1]
    cov = [[3,0], [0,3]]
    m, n= multivariate_normal(mean, cov, size=1000).T
    for i in range(len(m)):
        x.append([m[i], n[i]])
    y = [1]*1000 + [-1]*1000

    return np.array(x), np.array(y)


x_train, y_train = generate_data()
x_test, y_test = generate_data()
#i = 1
C_set = [0.01, 0.1, 1, 10, 100, 1000]
for c in C_set:
    #i += 1
    clf = SVC(C=c, kernel='rbf', gamma=1)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    plot_decision_regions(x_train, y_train, clf=clf, legend=2)
    #plt.figure(i)
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.title('C=' + str(c) + ', Accuracy=' + str(accuracy) + ', # of support vectors=' + str(clf.n_support_))

    plt.show()
