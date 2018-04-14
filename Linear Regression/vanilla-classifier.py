import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# ## Data: Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address, and their color Avatar. Then it also has numerical value columns:
# 
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member. 
# 


customers = pd.read_csv("Ecommerce Customers")

# Model Selection
X = customers[['Time on App', 'Time on Website','Avg. Session Length','Length of Membership']]
y = customers['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.4, random_state=101)
lm = LinearRegression()
lm.fit(X_train, y_train)


# Model Evaluation
mae = metrics.mean_absolute_error(y_test, pred);
mse = metrics.mean_squared_error(y_test, pred);
rmse = np.sqrt(mse)
