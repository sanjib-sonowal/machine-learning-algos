# 1. Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from config import DATASET_PATH

# 2. Loading data file
balance_data = pd.read_csv(DATASET_PATH + '/decision_tree_loan_repayment.csv',
                           sep=',', header=0)
print("Dataset: ", balance_data.head())
print("Dataset Length: ", len(balance_data))
print("Dataset Shape: ", balance_data.shape)

# 3. Separating the Target variables
X = balance_data.values[:, 0:4]
Y = balance_data.values[:, 5]

# 4. Splitting dataset in Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# 5. Function to perform training with Entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

# 6. Function to make Predictions
y_pred_en = clf_entropy.predict(X_test)
print(y_pred_en)

# 7. Checking Accuracy
print("Accuracy is: ", accuracy_score(y_test, y_pred_en)*100)
