"""
Desc: Group Crocodiles or Alligators
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets._samples_generator import make_blobs


# Create 40 separable points
X, y = make_blobs(n_samples=40, centers=2, random_state=20)

# Fit the model, don't regularized for illustration purpose
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

# Display the data in graph form
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
# plt.show()

# Using to predict unknown data
new_data = [[3, 4], [5, 6]]
print(clf.predict(new_data))

# Plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# Create grid to evaluate models
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margin
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
# Plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none')
plt.show()
