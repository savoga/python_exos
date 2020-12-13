'''
In this implementation, x has one dimension only.
x=[1,2,3,...,29,30] NEVER CHANGES
y_target=random numbers
F=y_est is updated at each loop
'''

import trees_from_scratch as tfs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(0,30)
x = pd.DataFrame({'x':x})

# just random uniform distributions in differnt range

y1 = np.random.uniform(10,15,10)
y2 = np.random.uniform(0,5,10)
y3 = np.random.uniform(13,17,10)

y_target = np.concatenate((y1,y2,y3))

plt.figure()
plt.plot(x,y_target, 'o')

n = x.shape[0]
p = x.shape[1]

# 1. Initial function = mean of y
F_0 = np.ones(n)*np.mean(y_target)
n_estimators = 1

plt.plot(x,F_0,'r')

F = F_0

for i in range(n_estimators):

    # 2. compute pseudo-residuals
    res = y_target - F_0
    res = res.reshape(len(res),1)

    # 3. train base learner
    tree = tfs.DecisionTree(max_depth=3)
    tree = tree.build_tree(np.concatenate([x, res], axis=1)) # we predict the residuals
    # TODO: replace with a tree that can handle REGRESSION
    split_element = tree.split_element.value

    left_idx = np.where(x <= split_element)[0]
    right_idx = np.where(x > split_element)[0]

    left_mean = np.array(x)[left_idx].mean()
    right_mean = np.array(x[right_idx]).mean()

    pred = np.ones((len(x),1))
    pred[left_idx] = left_mean
    pred[right_idx] = right_mean

    # 4. update function
    F = F + pred

    # plotting after prediction
    plt.figure()
    plt.plot(x,y_target, 'o')
    plt.plot(x, F)