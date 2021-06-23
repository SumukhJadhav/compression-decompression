import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from PIL import Image,ImageChops
from sklearn.utils import shuffle
from time import time

t1 = time()

n_colors =64

img  = Image.open("1.jpg")

img = np.array(img, dtype=np.float64) / 255

w, h, d = original_shape = tuple(img.shape)
assert d == 3
image_array = np.reshape(img, (w * h, d))


image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

print("Predicting colors...")
t0 = time()
labels = kmeans.predict(image_array)
print("Predicted in %0.3fs." % (time() - t0))


def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

plt.figure(2)
plt.clf()
plt.axis('off')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
figure = plt.gcf()
figure.set_size_inches(16, 12)
plt.savefig('2.jpg',bbox_inches='tight',pad_inches = 0,dpi =100)   
#plt.show()
print("Compressed in %0.3fs." % (time() - t1))
