'''
TODO:
    hasNonConvexCluster()
    we should do the test on all points that are between every two points
    (in terms of Norm 2 I think) of a cluster: if it is not inside the cluster, then it
    it non convex
    Limit: can be quite long
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.neighbors import NearestNeighbors

from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import Point, Polygon

# Color samples depending on labels
def show_samples(samples, labels, features = [0,1], feature_names = None, display_labels = True):
    '''Display the samples in 2D'''
    if display_labels:
        nb_labels = np.max(labels)
        for j in range(nb_labels + 1):
            nb_samples = np.sum(labels == j)
            if nb_samples:
                index = np.where(labels == j)[0]
                plt.scatter(samples[index,features[0]],samples[index,features[1]], label=str(j))
                plt.legend(loc="upper left")
    else:
        plt.scatter(samples[:,features[0]],samples[:,features[1]],color='gray')
    if feature_names is not None:
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
    # plt.axis('equal')
    # plt.show()

# Detect whether clusters are convex
def hasNonConvexCluster(data, labels):

    data_arr = np.array(data)
    labels_unique = np.unique(labels)

    for cluster in labels_unique:
        if cluster == -1:
            continue
        points = data_arr[np.where(labels==cluster)]
        # If number of points is too low to create a cluster
        # E.g. needs 4 points to create a polygon in dimension 4
        if(len(points)<data.shape[1]+1):
            print('Cluster {}: not enough points to create a polygon'.format(cluster))
            continue
        hull = ConvexHull(points) # impossible to draw flat polygons in 3D??
        # Display (2D only)
        # for simplex in hull.simplices:
        #     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

        polygon_idx = np.unique(hull.simplices.flatten())
        polygon_points = points[polygon_idx]
        polygon = Polygon(polygon_points)

        points_test = data_arr[np.where(labels!=cluster)]
        for p in points_test:
            point = Point(p)
            if(polygon.contains(point)):
                print('Cluster {} (label) is non convex!'.format(cluster))

# Simulate 3D ring; examples from
# https://stackoverflow.com/questions/42296761/masking-a-3d-numpy-array-with-a-tilted-disc
def ringCoordinates():
    space=np.zeros((40,40,20))

    r = 8 #radius of the circle
    theta = np.pi / 4 # "tilt" of the circle
    phirange = np.linspace(0, 3)
    # phirange = np.linspace(0, 2 * np.pi) #to make a full circle

    #center of the circle
    center=[20,20,10]

    #computing the values of the circle in spherical coordinates and converting them
    #back to cartesian
    for phi in phirange:
        x = r * np.cos(theta) * np.cos(phi) + center[0]
        y=  r*np.sin(phi)  + center[1]
        z=  r*np.sin(theta)* np.cos(phi) + center[2]
        space[int(round(x)),int(round(y)),int(round(z))]=1
    x,y,z = space.nonzero()

    # 2nd ring
    x2 = x - 10
    y2 = y
    z2 = z
    x = np.concatenate([x, x2])
    y = np.concatenate([y, y2])
    z = np.concatenate([z, z2])

    for i in range(x.shape[0]):
        if i % 2==0:
            x[i] = x[i] + 1
            z[i] = z[i] + 2

    return np.stack([x,y,z], axis=1)

# Parameter estimation for epsilon in DBSCAN
def dbscanEps(X, min_samples):
    n_neighbors = min_samples
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    k_dist = np.apply_along_axis(lambda x: x[n_neighbors-1], axis=1, arr=distances)
    k_dist_sorted = -np.sort(-k_dist)
    diff_extreme = abs(k_dist_sorted[0]-k_dist_sorted[len(k_dist_sorted)-1])/len(k_dist_sorted)
    k_dist_diff = abs(np.diff(k_dist_sorted))
    idx_threshold = (np.abs(k_dist_diff - diff_extreme)).argmin()
    plt.figure(figsize=(20, 8))
    plt.plot(np.arange(1,len(k_dist_sorted)+1), k_dist_sorted)
    plt.title('sorted k-dist graph')
    eps = k_dist_sorted[idx_threshold]
    plt.plot(np.arange(1,len(k_dist_sorted)+1)[idx_threshold], k_dist_sorted[idx_threshold], "o", color='r')
    return eps

#%%

#************ Test on fake 2D domain knoweldge dataset (convex clusters only) ************

df_2D = pd.DataFrame(columns=['Age', 'Balance'])

df_2D.loc[0] = [15, 500]
df_2D.loc[1] = [16, 800]
df_2D.loc[2] = [13, 700]

df_2D.loc[3] = [45, 3000]
df_2D.loc[4] = [43, 5085]
df_2D.loc[5] = [48, 2500]

df_2D.loc[6] = [63, 7600]
df_2D.loc[7] = [75, 8000]
df_2D.loc[8] = [73, 9000]

cluster = KMeans(n_clusters=3, random_state=0).fit(df_2D)
# cluster = DBSCAN(eps=1500, min_samples=2).fit(df_2D)

show_samples(np.array(df_2D), cluster.labels_, feature_names = ['Age','Balance'], display_labels = True)

#************ Test on fake 2D dataset (convex and non convex clusters) ************

X, y = make_moons(n_samples=100, shuffle=True, noise=None, random_state=None)
# X, y = make_blobs(n_samples=100, centers=3, n_features=2, cluster_std=1, random_state=39)

#cluster = KMeans(n_clusters=3, random_state=0).fit(X)

eps = 0.5
min_samples = X.shape[1]-1
cluster = DBSCAN(eps=eps, min_samples=2).fit(X)

y = cluster.labels_
fig = plt.figure()
show_samples(X, y, display_labels = True)
hasNonConvexCluster(X, y)

#%%

#************ Test on fake 3D dataset (convex and non convex clusters) ************

# GOOD 3D EXAMPLE FOR WHEN KMEANS DOESN'T WORK BUT DBSCAN DOES

X = ringCoordinates()

min_samples = X.shape[1]-1
eps = dbscanEps(X, min_samples)
cluster = DBSCAN(eps=3, min_samples=2).fit(X) # works well with double ring

# cluster = KMeans(n_clusters=3, random_state=0).fit(X)

y = cluster.labels_
# fig = plt.figure()
# show_samples(X, y, display_labels = True)
# hasNonConvexCluster(X, y)

fig = go.Figure(data=[go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2],
                                   mode='markers',
                                   marker=dict(
                                       size=12,
                                       color=cluster.labels_,
                                       colorscale='Viridis',
                                       opacity=0.8
                                       )
                                   )])
plot(fig)

#%%

#************ Fake 2D domain knowledge dataset (non convex) ************

df_2D = pd.DataFrame(columns=['Age', 'Balance'])

# Group 1
df_2D.loc[0] = [60, 950]
df_2D.loc[1] = [61, 945]
df_2D.loc[2] = [59, 940]

# Group 2
df_2D.loc[4] = [40, 950]
df_2D.loc[5] = [45, 960]
df_2D.loc[6] = [50, 970]
df_2D.loc[7] = [55, 980]
df_2D.loc[8] = [60, 980]
df_2D.loc[9] = [65, 970]
df_2D.loc[10] = [70, 960]
df_2D.loc[11] = [75, 950]
df_2D.loc[12] = [80, 940]

# cluster = KMeans(n_clusters=2, random_state=0).fit(df_2D)

# cluster = DBSCAN(eps=10, min_samples=2).fit(df_2D)

# to have two groups only, we can:
##### either increase epsilon:
cluster = DBSCAN(eps=12, min_samples=2).fit(df_2D)
# --> this would increase the neighborhood and thus transform all noise points
# in core/border points
##### or increase MinPts:
# cluster = DBSCAN(eps=10, min_samples=3).fit(df_2D)
# --> this would create more outliers since the constraint to be a cluster
# becomes stronger

labels = cluster.labels_
a = np.max(cluster.labels_)+1
labels = [a if x==-1 else x for x in cluster.labels_]
show_samples(np.array(df_2D), np.array(labels), feature_names = ['Age','Balance'], display_labels = True)
df_2D['Clusters'] = labels

hasNonConvexCluster(df_2D[['Age','Balance']], df_2D['Clusters'])

#%%
#************ Fake 3D domain knownledge dataset ************

df = pd.DataFrame(columns=['Age', 'Balance', 'Family'])

df.loc[0] = [15, 500, 0]
df.loc[1] = [16, 800, 0]
df.loc[2] = [13, 700, 1]

df.loc[3] = [45, 3000, 2]
df.loc[4] = [43, 5085, 3]
df.loc[5] = [48, 2500, 3]

df.loc[6] = [63, 7600, 4]
df.loc[7] = [75, 8000, 5]
df.loc[8] = [73, 9000, 6]

cluster = DBSCAN(eps=2000, min_samples=3).fit(df)

df['Clusters']=cluster.labels_

hasNonConvexCluster(df[['Age','Balance','Family']], df['Clusters'])

#************ Determine best parameters for DBSCAN ************

min_samples = 2 * df.shape[1] - 1
eps = dbscanEps(df, min_samples)
print(eps)

#%%
#************ Plot 3D ************

fig = go.Figure(data=[go.Scatter3d(x=df['Age'], y=df['Balance'], z=df['Family'],
                                   mode='markers',
                                   marker=dict(
                                       size=12,
                                       color=cluster.labels_,
                                       colorscale='Viridis',
                                       opacity=0.8
                                       )
                                   )])
plot(fig)

#************ HDBSCAN ************

import hdbscan

clusterer = hdbscan.RobustSingleLinkage(cut=0.125, k=7)
cluster_labels = clusterer.fit_predict(df)
hierarchy = clusterer.cluster_hierarchy_
alt_labels = hierarchy.get_clusters(0.100, 5)
hierarchy.plot()