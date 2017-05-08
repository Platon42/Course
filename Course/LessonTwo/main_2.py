import pandas as pn
import numpy as np
import sklearn.tree as st
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
plt.style.use('ggplot')

data = pn.read_csv('samples/titanic.csv')
selected_data = data[['Pclass', 'Fare', 'Age', 'Sex']]
y_data = data[['Survived']].apply(LabelEncoder().fit_transform)
x_data = selected_data.apply(LabelEncoder().fit_transform)

# not_null_data = data[data['Pclass'].notnull() & (data['Age'].notnull()) & (data['Fare'].notnull()) &
#                      (data['Survived'].notnull())]
# selected_data = not_null_data[['Pclass', 'Fare', 'Age', 'Sex']]
imp = sp.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp.fit(x_data,y_data)

# y_data = not_null_data[['Survived']].apply(LabelEncoder().fit_transform)
clf = st.DecisionTreeClassifier(random_state=241)
clf.fit(x_data, y_data)
feature_names = x_data.columns
importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]
d_first = 4
plt.figure(figsize=(8, 8))
plt.title("Feature importances")
plt.bar(range(d_first), importances[indices[:d_first]], align='center')
plt.xticks(range(d_first), np.array(feature_names)[indices[:d_first]], rotation=90)
plt.xlim([-1, d_first])
plt.show()
