import pandas as pn

import numpy as np
import sklearn.tree as st
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors, datasets
#from matplotlib.colors import ListedColormap


plt.style.use('ggplot')
svc = svm.SVC(C=1, kernel='linear')

data = pn.read_csv('samples/wine.csv',delimiter=',', sep='\t',
                   names=['Class','Alcohol','MalicAcid','Ash',
                          'AlcalinityOfAsh','Magnesium','TotalPhenols','Flavanoids','NonflavanoidPhenols',
                          'Proanthocyanins','ColorIntensity','Hue','OD280OD315OfDilutedWines','Proline'])
y = data[['Class']]

X = data[['Alcohol','MalicAcid','Ash','AlcalinityOfAsh','Magnesium','TotalPhenols','Flavanoids','NonflavanoidPhenols',
          'Proanthocyanins','ColorIntensity','Hue','OD280OD315OfDilutedWines','Proline']]

X_scale = sp.scale(X)
y_scale = sp.scale(y)
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.33, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

n_neighbors_array = []
for i in range(51):
    paths = i
    if i !=0:
        n_neighbors_array.append(paths)

knn = KNeighborsClassifier()

grid = GridSearchCV(knn, cv=kf, param_grid={'n_neighbors': n_neighbors_array})
grid.fit(X_train, y_train.values.ravel())

best_cv_err = 1 - grid.best_score_
best_n_neighbors = grid.best_estimator_.n_neighbors

scores = cross_val_score(svc, X_test, y_test.values.ravel(), cv=kf)

scores = cross_val_score(svc, X, y.values.ravel(), cv=kf)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print (best_cv_err, best_n_neighbors)
print(grid.best_score_)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
#
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
#
# h = .02  # step size in the mesh
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#
# n_neighbors_array = []
# for i in range(51):
#     paths = i
#     if i != 0:
#         n_neighbors_array.append(paths)
#
# for weights in ['uniform', 'distance']:
#     # we create an instance of Neighbours Classifier and fit the data.
#
#     knn = KNeighborsClassifier()
#     grid = GridSearchCV(knn, cv=kf, param_grid={'n_neighbors': n_neighbors_array})
#     grid.fit(X_train, y_train.values.ravel())
#
#     best_cv_err = 1 - grid.best_score_
#     best_n_neighbors = grid.best_estimator_.n_neighbors
#
#     clf = neighbors.KNeighborsClassifier(best_n_neighbors, weights=weights)
#     clf.fit(X, y)
#
#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, x_max]x[y_min, y_max].
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#
#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     plt.figure()
#     plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
#
#     # Plot also the training points
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     plt.title("3-Class classification (k = %i, weights = '%s')"
#               % (best_n_neighbors, weights))
#
# plt.show()

#
#
#
# y_train_predict = knn.predict(X_train)
# y_test_predict = knn.predict(X_test)
# print(y_test_predict)

# err_train = np.mean(y_train != y_train_predict)
# err_test = np.mean(y_test  != y_test_predict)
# print (err_train, err_test)

# knn = KNeighborsClassifier()
# for X_train, X_test in kf.split(X):
#     nn = KNeighborsClassifier()
#     knn.fit(X_train, y.values.ravel())
#     score = cross_val_score(svc, X_train, y, cv=kf, n_jobs=-1)
# print(score)
#     # X_train, X_test = X[train_index], X[test_index]
    # y_train, y_test = y[train_index], y[test_index]