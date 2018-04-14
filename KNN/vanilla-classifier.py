import pandas as pd
import numpy as  np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Data
dt = pd.read_csv("KNN_Project_Data")

# ** Create a StandardScaler() object called scaler.**

scaler = StandardScaler()
scaler.fit(dt.drop("TARGET CLASS", axis=1))
scaled_feats = scaler.transform(dt.drop("TARGET CLASS", axis=1))
df_feat = pd.DataFrame(scaled_feats, columns=dt.columns[:-1])

# # Model Selection
k=10
X=df_feat
y=dt["TARGET CLASS"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)


# # Model Evaluation
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# # Choosing a better K

err = []
for i in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    err.append(np.mean(pred!=y_test))

plt.plot(range(1,100), err)
plt.title("Error rate vs K")
plt.xlabel("K")
plt.ylabel("Error rate")


# ## Retrain with new K Value

knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
