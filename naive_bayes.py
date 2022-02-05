# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

sns.set()

data = fetch_20newsgroups()
# print(data.target_names)

# Defining all the categories
categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
              'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
              'talk.religion.misc']
# Training the data on these categories
train = fetch_20newsgroups(subset='train', categories=categories)
# Testing the data for these categories
test = fetch_20newsgroups(subset='test', categories=categories)

# Print train and test data
print("Total Articles: ", len(train.data))
# print(train.data[8])
# print(train.target[8])
# print(test.data[8])

# Creating a model based on Multinomial Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# Training the model with the train data
model.fit(train.data, train.target)
# Creating labels for test data
labels = model.predict(test.data)

# Creating confusion matrix and heat map
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=train.target_names,
            yticklabels=train.target_names)

# Plotting Heatmap of Confusion Matrix
plt.xlabel('true label')
plt.ylabel('predicted label')
# plt.show()


# Predicting category on new data based on trained model
def predict_category(s, p_train=train, p_model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


# Run prediction
print(predict_category('Albert Einstein'))
