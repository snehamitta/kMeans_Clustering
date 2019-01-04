from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas

data = pandas.read_csv('/Users/snehamitta/Desktop/ML/Assignment2/cars(1).csv', delimiter = ',')

trainData = data[["Horsepower", "Weight"]]

#3a) Elbow values 
wk = []
K = range(1,16)
for k in K:
    kmeanModel = KMeans(n_clusters = k, random_state = 0).fit(trainData)
    wk.append((kmeanModel.inertia_)/trainData.shape[0])
   
print(wk)
plt.plot(K, wk, 'bx-')
plt.xlabel('k')
plt.ylabel('Wk')
plt.show()

# Silhouette values 
s = []
K = range(2,16)
for k in K:
    kmeanModel = KMeans(n_clusters = k, random_state = 0).fit(trainData)
    labels = kmeanModel.fit_predict(trainData)
    s.append(metrics.silhouette_score(trainData, labels, metric = 'euclidean'))
    
print(s)
plt.plot(K,s, 'bx-')
plt.show()
    