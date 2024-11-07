import sklearn
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#Data for training

# traingingData = np.array(list(csv.reader(open('D1.csv'))))

# testingData = np.array(list(csv.reader(open('D2.csv'))))

dataFrame = pd.read_csv("D1.csv")

data = dataFrame.to_numpy()

train_y = data[:, 41]

train_data = data[:, 0:41]

# KNN

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(train_data, train_y)




newDataFrame = pd.read_csv("D2.csv")

data = newDataFrame.to_numpy()
    
answer = knn.predict(data)

with open("answer.txt", "w") as txt_file:
    for line in answer:
        txt_file.write(str(int(line)) + "\n")



# # Normalization

# train_normalized = normalize(train_data, norm='l2')


# # PCA

# pca = PCA(n_components=8)
# pca.fit(train_normalized)

# print(pca.components_)
        


# print("Import successful")