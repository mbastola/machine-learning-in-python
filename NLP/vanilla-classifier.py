import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Data: [Yelp Review Data Set from Kaggle](https://www.kaggle.com/c/yelp-recsys-2013).
yelp = pd.read_csv('yelp.csv')

#feature engineering

# **New column called "text length" is the number of words in the text column.**
yelp['text length']=yelp['text'].apply(len)
yelp_binary = yelp[(yelp['stars']==1)|(yelp['stars']==5)] 

X = yelp_binary['text']
y=yelp_binary['stars']
cvec = CountVectorizer()
X = cvec.fit_transform(X)

#Model selection: Multinomial Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# ## Model Evaluations
pred=classifier.predict(X_test)
print(confusion_matrix(y_test,pred))
print("\n")
print(classification_report(y_test,pred))


# ** Create a pipeline with the following steps:CountVectorizer(), TfidfTransformer(),RandomForestClassifier(200)/**

pipeline_mnb = Pipeline([('bow',CountVectorizer()),('tfidf',TfidfTransformer()),("classification",MultinomialNB())])


pipeline_rf = Pipeline([('bow',CountVectorizer()),('tfidf',TfidfTransformer()),("classification",RandomForestClassifier(200))])


# ## Using Pipeline

X = yelp_binary['text']
y=yelp_binary['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pipeline_mnb.fit(X_train,y_train)
pipeline_rf.fit(X_train,y_train)


# ### Models Evaluation
pred_tfidf = pipeline_rf.predict(X_test)
print(confusion_matrix(y_test,pred_tfidf))
print("\n")
print(classification_report(y_test,pred_tfidf))

pred_tfidf = pipeline_mnb.predict(X_test)
print(confusion_matrix(y_test,pred_tfidf))
print("\n")
print(classification_report(y_test,pred_tfidf))
