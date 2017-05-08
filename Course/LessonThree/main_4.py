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

from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston

svc = svm.SVC(C=1, kernel='linear')
boston = load_boston()

X = pn.DataFrame(boston.data)
y = pn.DataFrame(boston.target)

X_scale = sp.scale(X)
y_scale = sp.scale(y)
#X_train, X_test, y_train, y_test = train_test_split(X_scale, y_scale, test_size=0.33, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

neigh = KNeighborsRegressor(n_neighbors=5, weights='distance')
neigh.fit(X_scale, y_scale)

scores = cross_val_score(svc, X_scale, y_scale, cv=kf, scoring='neg_mean_squared_error')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
