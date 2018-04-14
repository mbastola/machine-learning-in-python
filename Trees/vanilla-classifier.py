import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

# Data: Lending data from 2007-2010 (https://www.lendingclub.com/info/download-data.action)

loans = pd.read_csv("loan_data.csv")
# Feature Engineering
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)

#Model Selection: Dtree
X = final_data.drop('not.fully.paid',axis = 1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

### Model Evaluation
pred = dtree.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

#Model Selection: RandomForest
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)

### Model Evaluation
pred_rf = rfc.predict(X_test)
print(classification_report(y_test,pred_rf))
print(confusion_matrix(y_test,pred_rf))
