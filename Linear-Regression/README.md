[This project](https://github.com/mbastola/machine-learning-in-python/edit/master/Linear-Regression) comprised of Linear Regression in python.

<div>
** Data head: **
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Email</th>
      <th>Address</th>
      <th>Avatar</th>
      <th>Avg. Session Length</th>
      <th>Time on App</th>
      <th>Time on Website</th>
      <th>Length of Membership</th>
      <th>Yearly Amount Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mstephenson@fernandez.com</td>
      <td>835 Frank Tunnel\nWrightmouth, MI 82180-9605</td>
      <td>Violet</td>
      <td>34.497268</td>
      <td>12.655651</td>
      <td>39.577668</td>
      <td>4.082621</td>
      <td>587.951054</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hduke@hotmail.com</td>
      <td>4547 Archer Common\nDiazchester, CA 06566-8576</td>
      <td>DarkGreen</td>
      <td>31.926272</td>
      <td>11.109461</td>
      <td>37.268959</td>
      <td>2.664034</td>
      <td>392.204933</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pallen@yahoo.com</td>
      <td>24645 Valerie Unions Suite 582\nCobbborough, D...</td>
      <td>Bisque</td>
      <td>33.000915</td>
      <td>11.330278</td>
      <td>37.110597</td>
      <td>4.104543</td>
      <td>487.547505</td>
    </tr>
    <tr>
      <th>3</th>
      <td>riverarebecca@gmail.com</td>
      <td>1414 David Throughway\nPort Jason, OH 22070-1220</td>
      <td>SaddleBrown</td>
      <td>34.305557</td>
      <td>13.717514</td>
      <td>36.721283</td>
      <td>3.120179</td>
      <td>581.852344</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mstephens@davidson-herman.com</td>
      <td>14023 Rodriguez Passage\nPort Jacobville, PR 3...</td>
      <td>MediumAquaMarine</td>
      <td>33.330673</td>
      <td>12.795189</td>
      <td>37.536653</td>
      <td>4.446308</td>
      <td>599.406092</td>
    </tr>
  </tbody>
</table>
</div>



<div>
** Data crude metrics: **
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Avg. Session Length</th>
      <th>Time on App</th>
      <th>Time on Website</th>
      <th>Length of Membership</th>
      <th>Yearly Amount Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>33.053194</td>
      <td>12.052488</td>
      <td>37.060445</td>
      <td>3.533462</td>
      <td>499.314038</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.992563</td>
      <td>0.994216</td>
      <td>1.010489</td>
      <td>0.999278</td>
      <td>79.314782</td>
    </tr>
    <tr>
      <th>min</th>
      <td>29.532429</td>
      <td>8.508152</td>
      <td>33.913847</td>
      <td>0.269901</td>
      <td>256.670582</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.341822</td>
      <td>11.388153</td>
      <td>36.349257</td>
      <td>2.930450</td>
      <td>445.038277</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>33.082008</td>
      <td>11.983231</td>
      <td>37.069367</td>
      <td>3.533975</td>
      <td>498.887875</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>33.711985</td>
      <td>12.753850</td>
      <td>37.716432</td>
      <td>4.126502</td>
      <td>549.313828</td>
    </tr>
    <tr>
      <th>max</th>
      <td>36.139662</td>
      <td>15.126994</td>
      <td>40.005182</td>
      <td>6.922689</td>
      <td>765.518462</td>
    </tr>
  </tbody>
</table>
</div>


    RangeIndex: 500 entries, 0 to 499
    Data columns (total 8 columns):
    Email                   500 non-null object
    Address                 500 non-null object
    Avatar                  500 non-null object
    Avg. Session Length     500 non-null float64
    Time on App             500 non-null float64
    Time on Website         500 non-null float64
    Length of Membership    500 non-null float64
    Yearly Amount Spent     500 non-null float64
    dtypes: float64(5), object(3)
    memory usage: 31.3+ KB


## Data Exploration



    <seaborn.axisgrid.JointGrid at 0x7ffa895f8a10>




![png](https://github.com/mbastola/machine-learning-in-python/blob/master/Linear-Regression/output_10_2.png)



    <matplotlib.axes._subplots.AxesSubplot at 0x7ffaac938f50>




![png](https://github.com/mbastola/machine-learning-in-python/blob/master/Linear-Regression/output_11_1.png)


** with the Time on App column instead. **


    <seaborn.axisgrid.JointGrid at 0x7ffaae6a5d10>




![png](https://github.com/mbastola/machine-learning-in-python/blob/master/Linear-Regression/output_13_1.png)


** jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**



    <seaborn.axisgrid.JointGrid at 0x7ffaae5a0490>




![png](https://github.com/mbastola/machine-learning-in-python/blob/master/Linear-Regression/output_15_1.png)


**types of relationships across the entire data set.**

    <seaborn.axisgrid.PairGrid at 0x7ffaae5f8e90>




![png](https://github.com/mbastola/machine-learning-in-python/blob/master/Linear-Regression/output_17_1.png)



**linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership. **


    <seaborn.axisgrid.FacetGrid at 0x7ffaacef4bd0>




![png](https://github.com/mbastola/machine-learning-in-python/blob/master/Linear-Regression/output_20_1.png)


**coefficients of the model**



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
      <th>Coeff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Time on App</th>
      <td>37.892600</td>
    </tr>
    <tr>
      <th>Time on Website</th>
      <td>0.560581</td>
    </tr>
    <tr>
      <th>Avg. Session Length</th>
      <td>25.691540</td>
    </tr>
    <tr>
      <th>Length of Membership</th>
      <td>61.648594</td>
    </tr>
  </tbody>
</table>
</div>



## Predicting Test Data
    <matplotlib.collections.PathCollection at 0x7ffa8405d3d0>




![png](https://github.com/mbastola/machine-learning-in-python/blob/master/Linear-Regression/output_33_1.png)


** scatterplot of the real test values versus the predicted values. **



    <matplotlib.text.Text at 0x135546320>




![png](https://github.com/mbastola/machine-learning-in-python/blob/master/Linear-Regression/output_35_1.png)


## Model Evaluation

Model performance wrt residual sum of squares and the explained variance score (R^2).

** Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.**


mae : 7.74267128583874
mse : 93.83297800820083
rmse : 9.686742383701594





## Residuals


    <matplotlib.axes._subplots.AxesSubplot at 0x7ffa7d1a0590>




![png](https://github.com/mbastola/machine-learning-in-python/blob/master/Linear-Regression/output_39_1.png)


## Conclusion
We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coeffecient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Avg. Session Length</th>
      <td>25.981550</td>
    </tr>
    <tr>
      <th>Time on App</th>
      <td>38.590159</td>
    </tr>
    <tr>
      <th>Time on Website</th>
      <td>0.190405</td>
    </tr>
    <tr>
      <th>Length of Membership</th>
      <td>61.279097</td>
    </tr>
  </tbody>
</table>
</div>



** How can we interpret these coefficients? **

**should the company focus more on their mobile app or on their website?**

Data shows Mobile App since time on App shows to have a larger slope wrt yearly amount spent.


