# Loading the data with the iris dataset
from sklearn.datasets import load_iris
# Loading scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Setting random seed
np.random.seed(0)

# Creating an object iris with the iris data
iris = load_iris()
# Creating a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# Viewing the top 5
print(df.head())

# Adding a new column for the species name
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
# Viewing the top 5
print(df.head())

# Creating Test and Train data
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
# Viewing the top 5
print(df.head())

# Creating dataframes with test rows and training rows
train, test = df[df['is_train'] == True], df[df['is_train'] == False]

# Show the number of observations for the test and training dataframes
print('Number of observations in training data:', len(train))
print('Number of observations in test data:', len(test))

# Creating a list of the feature column's names
features = df.columns[:4]
# View features
print(features)

# Converting each species name into digits
y = pd.factorize(train['species'])[0]
# Viewing target
print(y)

# Creating a random forest Classifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Training the classifier
clf.fit(train[features], y)

# Applying the trained Classifier to the test
print(test[features])
clf.predict(test[features])

# Viewing the predicted probabilities of the first 10 observations
clf.predict_proba(test[features])[0:10]

# Mapping names for the plants for each of the predicted plant class
preds = iris.target_names[clf.predict(test[features])]

# View the Predicted species of the first five observations
print(preds[0:25])

# View the Actual species of the first five observations
print(test['species'].head())

# Creating confusion matrix
print(pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species']))

# preds = iris.target_names[clf.predict(test[features])]
preds = iris.target_names[clf.predict([[5.0, 3.6, 1.4, 2.0], [5.0, 3.6, 1.4, 2.0]])]
print(preds)
