import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

# ## The Data
# [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set). 
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

iris = sns.load_dataset('iris')

## # Model Selection
X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
svc_model =  SVC()
svc_model.fit(X_train, y_train)


# ## Model Evaluation
pred = svc_model.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred))

#Using Gridsearch for better SVM parameters
param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001]}
grid = GridSearchCV(SVC(),param_grid, verbose=2)

grid.fit(X_train,y_train)

grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
