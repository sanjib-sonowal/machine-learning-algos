# 1. Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 2. Load the data
digits = load_digits()
print("Image data shape", digits.data.shape)
print("Label data shape", digits.target.shape)

# 3. Plot the dataset
plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index+1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title("Training %i\n" % label, fontsize=20)
# plt.show()

# 4. Splitting the data into Train and Test set
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state=2)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# 5. Fitting Logistic regression to the Training set
logisticReg = LogisticRegression()
logisticReg.fit(x_train, y_train)

# 6. Predict for one observation
print(logisticReg.predict(x_test[0].reshape(1, -1)))
print(logisticReg.predict(x_test[0:10]))
predictions = logisticReg.predict(x_test)

# 7. Get the predicted score
score = logisticReg.score(x_test, y_test)
print("Score: ", score)

# 8. Create a confusion matrix
cm = metrics.confusion_matrix(y_test, predictions)
print("Confusion Matrix: ", cm)

# 9. Visualize the data
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap="Blues_r")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
all_sample_title = "Accuracy Score: {0}".format(score)
plt.title(all_sample_title, size=15)

index = 0
misclassifiedIndex = []
for predict, actual in zip(predictions, y_test):
    if predict == actual:
        misclassifiedIndex.append(index)
    index += 1
plt.figure(figsize=(20, 3))
for plotIndex, wrong in enumerate(misclassifiedIndex[0:4]):
    plt.subplot(1, 4, plotIndex + 1)
    plt.imshow(np.reshape(x_test[wrong], (8, 8)), cmap=plt.cm.gray)
    plt.title("Predicted: {}, Actual: {}".format(predictions[wrong], y_test[wrong]), fontsize=10)
plt.show()
