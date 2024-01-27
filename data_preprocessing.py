import numpy as np
from sklearn.model_selection import train_test_split

# Load and split data into training, validation and test sets
def load_and_split_data(data_dir, test_size=0.2, random_state=None):
    try:
        # Load training data
        train_images = np.load(f'{data_dir}/train_images.npy')
        train_labels = np.load(f'{data_dir}/train_labels.npy')

        # Load test data
        test_images = np.load(f'{data_dir}/test_images.npy')
        test_labels = np.load(f'{data_dir}/test_labels.npy')

        # Split training data into training and validation
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images, train_labels, test_size=test_size, random_state=random_state)

        return train_images, val_images, train_labels, val_labels, test_images, test_labels

    except FileNotFoundError:
        print(f"Error: Data files not found in {data_dir}")
        return None, None, None, None, None, None
