import urllib.request
import gzip
import numpy as np
import os

def download_and_load_mnist_dataset(url, filename):
    urllib.request.urlretrieve(url + filename, filename)

    with gzip.open(filename, 'rb') as f:
        if 'images' in filename:
            data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        else:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

url = 'http://yann.lecun.com/exdb/mnist/'

train_images = download_and_load_mnist_dataset(url, 'train-images-idx3-ubyte.gz')
train_labels = download_and_load_mnist_dataset(url, 'train-labels-idx1-ubyte.gz')
test_images = download_and_load_mnist_dataset(url, 't10k-images-idx3-ubyte.gz')
test_labels = download_and_load_mnist_dataset(url, 't10k-labels-idx1-ubyte.gz')

# Create a dataset directory if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Save the data to .npy files in the dataset directory
np.save('dataset/train_images.npy', train_images)
np.save('dataset/train_labels.npy', train_labels)
np.save('dataset/test_images.npy', test_images)
np.save('dataset/test_labels.npy', test_labels)

# Delete the downloaded gzip files
os.remove('train-images-idx3-ubyte.gz')
os.remove('train-labels-idx1-ubyte.gz')
os.remove('t10k-images-idx3-ubyte.gz')
os.remove('t10k-labels-idx1-ubyte.gz')