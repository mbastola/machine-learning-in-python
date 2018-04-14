import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


ad_data = pd.read_csv("advertising.csv")

#Model Selection
X=ad_data[["Age","Daily Time Spent on Site","Area Income","Daily Internet Usage"]]
y=ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lm = LogisticRegression()
lm.fit(X_train,y_train)


# Model Evaluations
pred = lm.predict(X_test)
print(metrics.classification_report(y_test, pred))
print(metrics.confusion_matrix(y_test,pred))
