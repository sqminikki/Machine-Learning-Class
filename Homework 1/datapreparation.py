import numpy as np #This is just NumPy. It will making writing this type of code much easier.
from tensorflow.keras.datasets.mnist import mnist

# Extract digits 0, 1, and 9 ONLY.

# Vectorize each image into a column vector of length 784 and stack all vectors as columns of the data matrix 
# where N is the total number of selected images. Set the data type of X to float32.