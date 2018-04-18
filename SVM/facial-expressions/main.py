import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

def main():
    filename = "fer2013/fer2013.csv";
    df = pd.read_csv(filename)
    print(df.head())
    sns.countplot(x='emotion',data=df)

    binary = True  #read labels 3 and 4 only 
    if binary:
        df = df[(df['emotion']==3) | (df['emotion']==0)]
        df['emotion'] = df['emotion'].apply(lambda x: 0 if x == 0 else 1)
    X = df['pixels']
    y = df['emotion']

    sns.countplot(x=y)


    def splitFloat(x):
        splitted = x.split(" ")
        i = 0
        for item in splitted:
            splitted[i]=float(item)
            i=i+1
        return splitted

    def parseImageInput(X,y):
        X = X.apply(lambda x: splitFloat(x))
        X = np.array(X)
        X = np.stack(X,axis=0)
        y = np.array(y)
        return X,y


    X,y = parseImageInput(X,y)


    scaler = StandardScaler()
    scaler.fit(X)
    scaled_X = scaler.transform(X)


    pca = PCA(n_components=20)
    pca.fit(scaled_X)
    scaled_X_pca = pca.transform(scaled_X)

    plt.plot(pca.explained_variance_ratio_)

    plt.plot(pca.singular_values_)

    X_train, X_test, y_train, y_test = train_test_split(scaled_X_pca, y, test_size=0.2, random_state=42)

    plt.figure(figsize=(8,6))
    plt.scatter(scaled_X_pca[:,0],scaled_X_pca[:,1],c=y)

    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    error = np.mean(y_test != pred)
    print(error)


    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test, pred))

    param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001]}
    grid = GridSearchCV(SVC(),param_grid, verbose=2)

    grid.fit(X_train,y_train)

    pred = grid.predict(X_test)

    error = np.mean(y_test != pred)
    print(error)
    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test,pred))

if __name__ == "__main__":
    main()