import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import sys
from numpy import linalg
from skimage import data
from skimage.color import rgb2gray
from PIL import Image

# Create our data matrix... we assume from the beginning that there
# are only 25 faces total. We also assume the dimensions of each face
# will be 172 x 180. That's my choice. You can make your own.
h = 90
w = 86

# Read in each of the images from file, and resize.
X = np.zeros((1,h*w))
A = np.zeros((w,h))

for i in range(5):
	for j in range(1,6):
		image_no = str(i*5 + j)
		print("Reading " + image_no + ".jpg...")
		face      = Image.open("Training_Data/" + image_no + ".jpg")

		# Shrink the image and convert to grayscale.
		face      = face.resize((h,w))
		gray_face = rgb2gray(face)

		# Average face.
		A = A + gray_face

		# Edge case where we don't need to append.
		if (i == 0 and j == 1) :
			X = gray_face.reshape((1,h*w))
		else:
			X = np.append(X,gray_face.reshape((1,h*w)), axis=0)

# Average face, normalized by number of photos.
A       = A / 25.0
avgFace = A.reshape((1,h*w))

# Subtract off the average from the data matrix.

for i in range(X.shape[0]):
	X[i,:] = X[i,:] - avgFace

# Print the average face to a jpg image.
plt.imsave('avg_face.png', A, cmap='gray')

# Compute correlation/covariance matrix:
C = np.matmul(X.T,X)

# Get eigenvectors and eigenvalues.
print('Performing the SVD...')
U,S,V = np.linalg.svd(C,full_matrices=1)

# Get the sum of the eigenvalues to compute the %variance captured.
S_sum = 0.0
for i in range(S.shape[0]):
	S_sum = S_sum + S[i]

# Print out the eigenfaces.
for i in range(U.shape[1]):
	if (S[i] > 1e-2):
		eig_face = U[:,i].reshape((h,w), order='F')
		print("Writing eigenvector " + str(i) + " with eigenvalue " + str(S[i]) + " (%var: " + str(S[i]/S_sum) + ")")
		plt.imsave(str(i) + "_eig.png", eig_face, cmap='gray')


# Print out the singular values.
S = S / S_sum
C_sum = 0.0
for i in range(S.shape[0]):
    C_sum = C_sum + S[i]
    if (S[i] > 1e-2):
        print(str(i) + " " + str(S[i]) + " " + str(C_sum))

