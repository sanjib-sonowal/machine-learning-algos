# 1. Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 2. Importing the dataset and Extracting the Independent and Dependent variables
companies = pd.read_csv('E:/Projects/Machine-Deep-Learning/_my_hands_on/datasets/1000_Companies.csv')
X = companies.iloc[:, :-1].values
y = companies.iloc[:, 4].values

companies.head()
# print(companies.head())

# 3. Data Visualisation
# Building the Correlation matrix
sns.heatmap(companies.corr())
# plt.show()

# 4. Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
label_encoder = LabelEncoder()
X[:, 3] = label_encoder.fit_transform(X[:, 3])
one_hot_encoder = ColumnTransformer(
    [('OHE', OneHotEncoder(), [3])],
    remainder='passthrough'
)
X = one_hot_encoder.fit_transform(X)
print(X)

# 5. Avoiding the Dummy variable trap
X = X[:, 1:]

# 6. Splitting the data into Train and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 7. Fitting Multiple Linear regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 8. Predicting the Test set results
y_pred = regressor.predict(X_test)
# print(y_pred)

# 9. Calculating the Coefficients and Intercepts
print(regressor.coef_)
print(regressor.intercept_)

# 10. Calculating the R squared value
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
print(r2_score(y_test, y_pred))
