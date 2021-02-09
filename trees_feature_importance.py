import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

data = pd.DataFrame(data=[
                     [2,7,1],
                     [2,1,2],
                     [1,8,2],
                     [12,2,1],
                     [12,2,8],
                     ],
                    columns=[
                         'feat1',
                         'feat2',
                         'label'
                         ])

X = data[['feat1','feat2']]
y = data['label']

clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X,y)
fig = plt.figure(figsize=(10,10))
print(round(accuracy_score(clf.predict(X),y)*100,2))
tree.plot_tree(clf)

print(clf.feature_importances_)
ni_0_0 = 0.64-0.5*2/5-0.444*3/5
print(ni_0_0)
ni_0_1 = 3/5*(0.444-0.5*2/3)
print(ni_0_1)
ni_1 = 2/5*0.5
print(ni_1)

feat_importance_0 = (ni_0_0 + ni_0_1)/(ni_0_0 + ni_0_1 + ni_1)
print(feat_importance_0)
feat_importance_1 = (ni_1)/(ni_0_0 + ni_0_1 + ni_1)
print(feat_importance_1)
