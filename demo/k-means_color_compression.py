"""
Desc: Compress an image
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import load_sample_image
from sklearn.cluster import MiniBatchKMeans

# Fix numpy issues
warnings.simplefilter('ignore')

# Display a sample flower
flower = load_sample_image("flower.jpg")
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(flower)
# plt.show()

# Return the dimension of the array
print(flower.shape)

# Reshape the data to [n_samples X n_features] and rescale the color so they lie between 0 and 1
data = flower / 255.0
data = data.reshape(427 * 640, 3)
print(data.shape)


# Visualize these pixels in this color space using a subset of 10000 pixels of frequency
def plot_pixels(_data, title, colors=None, N=10000):
    if colors is None:
        colors = _data
    # Choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = _data[i].T

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel="Red", ylabel="Green", xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel="Red", ylabel="Blue", xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20)


plot_pixels(data, title="Input color space: 16 million possible colors")

kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

plot_pixels(data, colors=new_colors, title="Reduced color space: 16 colors")

flower_recolored = new_colors.reshape(flower.shape)
fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(flower)
ax[0].set_title("Original Image", size=16)
ax[1].imshow(flower_recolored)
ax[1].set_title("16 Color Image", size=16)

plt.show()
