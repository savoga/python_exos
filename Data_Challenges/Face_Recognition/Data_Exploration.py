import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import NearestNeighbors
import time

YTRAIN_PATH='../../../data_challenge/ytrain_challenge.csv'
XTRAIN_PATH='../../../data_challenge/xtrain_challenge.csv'
XTEST_PATH='../../../data_challenge/xtest_challenge.csv'

y = pd.read_csv(YTRAIN_PATH, nrows=10000)
x = pd.read_csv(XTRAIN_PATH, delimiter=',', nrows=10000)
xt = pd.read_csv(XTEST_PATH, delimiter=',', nrows=10000)

# Display of labels
randomListOfIndexes = [random.randint(0, int(len(y)/1000)) for iter in range(6)]

plt.figure(figsize=(15, 3))
fig, axs = plt.subplots(6, figsize=(15, 7))
fig.subplots_adjust(hspace=.5)
for idx, value in enumerate(randomListOfIndexes):
    axs[idx].plot(y[value*1000:(value+1)*1000])

# Display of distributions
for idx, colName in enumerate(x.columns):

    plt.figure(figsize=(10, 3))
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    axs[0].hist(x[colName], 20)
    axs[0].set_title(colName + " - train")

    axs[1].hist(xt[colName], 20)
    axs[1].set_title(colName + " - test")

# Display of main statistics
trainMeans = []
testMeans = []
trainStd = []
testStd = []
for colName in x.columns:
    trainMeans.append(np.mean(x[colName]))
    testMeans.append(np.mean(xt[colName]))
    trainStd.append(np.std(x[colName]))
    testStd.append(np.std(xt[colName]))


plt.figure(figsize=(15, 3))
plt.plot(trainMeans)
plt.plot(testMeans)
plt.title('means of features')

plt.figure(figsize=(15, 3))
plt.plot(trainStd)
plt.plot(testStd)
plt.title('std of features')

# *** Nearest neighbors ***
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(np.array(x))

# Test with a random value (to check the nearest neighbor is closer)
xt_one = np.array(xt)[0]
xtrain_simil_idx=neigh.kneighbors(xt_one.reshape(1,-1), return_distance=False).ravel()[0]
x_simil_nn = np.array(x)[xtrain_simil_idx]

x_simil_test = np.array(x)[5]

plt.figure(figsize=(15, 3))
plt.plot(xt_one)
plt.plot(x_simil_test)

plt.figure(figsize=(15, 3))
plt.plot(xt_one)
plt.plot(x_simil_nn)

# Find the best sample
list_simil_idx = xt.apply(lambda x: neigh.kneighbors(np.array(x).reshape(1,-1), return_distance=False).ravel()[0], axis=1)

x_simil = np.array(x)[list_simil_idx]
y_simil = np.array(y)[list_simil_idx]

def findBestSamplePandas(x, y, xt):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(np.array(x))
    list_simil_idx = xt.apply(lambda x: neigh.kneighbors(np.array(x).reshape(1,-1), return_distance=False).ravel()[0], axis=1)

    x_simil = np.array(x)[list_simil_idx]
    y_simil = np.array(y)[list_simil_idx]
    return x_simil, y_simil

def apply_nn_2(x_row):
    return neigh.kneighbors(x_row.reshape(1,-1), return_distance=False).ravel()[0]

start_time = time.time()
list_simil_idx_3 = np.apply_along_axis(apply_nn_2, 1, xt)
print("{} seconds".format(round(time.time() - start_time,2)))

def findBestSampleNumpy(x, y, xt):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(np.array(x))
    list_simil_idx = np.apply_along_axis(apply_nn_2, 1, xt)

    x_simil = np.array(x)[list_simil_idx]
    y_simil = np.array(y)[list_simil_idx]
    return x_simil, y_simil

np.savetxt('test-array.csv', list_simil_idx_3, fmt = '%i', delimiter=',')
test_array = pd.read_csv('test-array.csv', header=None)




from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

start_time = time.time()
idx_list_2=[]
def bestSample(xt):
    try:
        idx = neigh.kneighbors(xt.reshape(1,-1), return_distance=False).ravel()[0]
        idx_list_2.append(idx)
    except:
        print("error")
    return 0
with ThreadPoolExecutor(max_workers = 16) as executor:
    results = executor.map(bestSample, np.array(xt))
print("{} seconds".format(round(time.time() - start_time,2)))

