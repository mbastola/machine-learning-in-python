In [this project](https://github.com/mbastola/machine-learning-in-python/tree/master/KNN) we try out the most common Nearest neighbors technique: K-NN.
 


**Data Info**

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 11 columns):
    XVPM            1000 non-null float64
    GWYH            1000 non-null float64
    TRAT            1000 non-null float64
    TLLZ            1000 non-null float64
    IGGA            1000 non-null float64
    HYKR            1000 non-null float64
    EDFS            1000 non-null float64
    GUUB            1000 non-null float64
    MGJM            1000 non-null float64
    JHZC            1000 non-null float64
    TARGET CLASS    1000 non-null int64
    dtypes: float64(10), int64(1)
    memory usage: 86.0 KB


# Data Exploration

Since this data is artificial, we'll just do a large pairplot with seaborn.

**Seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.**


    <seaborn.axisgrid.PairGrid at 0x7f35caa4ac50>




![png](https://github.com/mbastola/machine-learning-in-python/blob/master/KNN/output_8_1.png)


**Converted the scaled features to a dataframe **

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>XVPM</th>
      <th>GWYH</th>
      <th>TRAT</th>
      <th>TLLZ</th>
      <th>IGGA</th>
      <th>HYKR</th>
      <th>EDFS</th>
      <th>GUUB</th>
      <th>MGJM</th>
      <th>JHZC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.568522</td>
      <td>-0.443435</td>
      <td>1.619808</td>
      <td>-0.958255</td>
      <td>-1.128481</td>
      <td>0.138336</td>
      <td>0.980493</td>
      <td>-0.932794</td>
      <td>1.008313</td>
      <td>-1.069627</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.112376</td>
      <td>-1.056574</td>
      <td>1.741918</td>
      <td>-1.504220</td>
      <td>0.640009</td>
      <td>1.081552</td>
      <td>-1.182663</td>
      <td>-0.461864</td>
      <td>0.258321</td>
      <td>-1.041546</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.660647</td>
      <td>-0.436981</td>
      <td>0.775793</td>
      <td>0.213394</td>
      <td>-0.053171</td>
      <td>2.030872</td>
      <td>-1.240707</td>
      <td>1.149298</td>
      <td>2.184784</td>
      <td>0.342811</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.011533</td>
      <td>0.191324</td>
      <td>-1.433473</td>
      <td>-0.100053</td>
      <td>-1.507223</td>
      <td>-1.753632</td>
      <td>-1.183561</td>
      <td>-0.888557</td>
      <td>0.162310</td>
      <td>-0.002793</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.099059</td>
      <td>0.820815</td>
      <td>-0.904346</td>
      <td>1.609015</td>
      <td>-0.282065</td>
      <td>-0.365099</td>
      <td>-1.095644</td>
      <td>0.391419</td>
      <td>-1.365603</td>
      <td>0.787762</td>
    </tr>
  </tbody>
</table>
</div>
# Predictions and Evaluations

** Confusion matrix and classification report.**

    [[109  43]
     [ 41 107]]



                 precision    recall  f1-score   support
    
              0       0.73      0.72      0.72       152
              1       0.71      0.72      0.72       148
    
    avg / total       0.72      0.72      0.72       300
    


# Choosing a better K Value



    Text(0,0.5,'Error rate')




![png](https://github.com/mbastola/machine-learning-in-python/blob/master/KNN/output_37_1.png)


## Retrained with new K Value

    [[124  28]
     [ 24 124]]
    
    
                 precision    recall  f1-score   support
    
              0       0.84      0.82      0.83       152
              1       0.82      0.84      0.83       148
    
    avg / total       0.83      0.83      0.83       300
    

