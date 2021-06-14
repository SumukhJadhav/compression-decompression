from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['figure.figsize'] = [16,8]

A = imread('scene.jpg')
X = np.mean(A, -1); #Converts RGB to greyscale

img = plt.imshow(X) #256 - X
img.set_cmap('gray')
plt.axis('off')
#plt.show()

U, S, VT = np.linalg.svd(X, full_matrices=False)
S = np.diag(S)
j = 0
r = 100

Xapprox = U[:,:r][:,:r] @ S[0:r,:r][0:r,:r] @ VT[:r,:][:r,:]
plt.figure(j+1)
j += 1
img = plt.imshow(Xapprox)
img.set_cmap('gray')
plt.axis('off')
plt.title('r= '+ str(r))
plt.show()

plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Singular values')
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('Singular values: Cumulative Sum')
plt.show()
