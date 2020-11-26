import trees_from_scratch as tfs
import numpy as np


x1 = np.random.normal(0,1,100)
x2 = np.random.normal(0,1,100)
x3 = np.random.normal(0,1,100)
x = np.stack([x1,x2,x3], axis=1)

y = np.random.randint(0,10,100)

n = x.shape[0]
p = x.shape[1]

# 1. Initial function = mean of y
F_0 = np.ones(n)*np.mean(y)
n_estimators = 10

for i in range(n_estimators):

    # 2. compute pseudo-residuals
    res = F_0 - y
    res = res.reshape(len(res),1)

    # 3. train base learner
    tree = tfs.build_tree(np.concatenate([x, res], axis=1))

    # 4. find gradient step
    # derive loss function?

    # 5. update function