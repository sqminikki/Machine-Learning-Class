
import numpy as np      # This is just NumPy. It will making writing this type of code much easier.
from tensorflow.keras.datasets import mnist

# Loat MNIST:
(x_train, y_train), _ = mnist.load_data() 

# Extract digits 0, 1, and 9 ONLY.
mask = np.isin(y_train, [0, 1, 9])
x = x_train[mask]   # Images
y = y_train[mask]   # Labels

"""
Vectorize each image into a column vector of length 784 and stack all vectors as columns of the data matrix 
 where N is the total number of selected images. Set the data type of X to float32. 
"""

num_images_N = x.shape[0]       # Number of selected images, N.
X = x.reshape(num_images_N, 784).T    # <--- Initial image with transpose. The ".T" means we transposed.
X = X.astype(np.float32)    # This sets the data type of X into float32.

# Now here comes the computational part:

# Mean global iamge:
myu = np.mean(X, axis = 1, keepdims = True)    # Note: the "keepdims" parameter tells NumPy to keep the reduced dimension as size 1. 

# Centered data matrix:
X_tilde = X - myu

# --- Sanity checks. Will be commended out when not needed. This is for confirmation.---
print("X shape:", X.shape)            
print("myu shape:", myu.shape)            
print("Centered mean (≈0):", X_tilde.mean())

