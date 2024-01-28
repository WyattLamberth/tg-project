import numpy as np
import tinygrad as tg
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn import Layer, Linear, Conv2D, MaxPool2D, Flatten, Dropout, ReLU, Sequential
from tinygrad.nn.optim import SGD

from data_preprocessing import *

class Aeolus:
    def __init__(self):
        self.l1 = Linear(784, 128, bias=False)
        self.l2 = Linear(128, 10, bias=False)

    def __call__(self, x):
        x = self.l1(x)
        x = x.leakyrelu()
        x = self.l2(x)
        return x
    
def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
    loss_mask = Y != ignore_index
    y_counter = Tensor.arange(self.shape[-1], dtype=dtypes.int32, requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    return self.log_softmax().mul(y).sum() / loss_mask.sum()
    
model = Aeolus()

# modify layers, activation functions, etc. here
data_dir = "~/Developer/tg-project/dataset"
train_images, val_images, train_labels, val_labels, test_images, test_labels = load_and_split_data(data_dir, test_size=0.2, random_state=42)

# print model summary
model.summary()

# loss function and optimizer
loss_fn = tg.losses.CrossEntropyLoss()
optimizer = SGD([model.l1.weight, model.l2.weight], lr=3e-4)

# training parameters
epochs = 10
batch_size = 32

# function to get batches of data
def get_batches(data, labels, batch_size):
    for start in range(0, len(data), batch_size):
        end = start + batch_size
        yield data[start:end], labels[start:end]

for epoch in range(epochs):
    model.train()
    for batch_data, batch_labels in get_batches(train_images, train_labels, batch_size):

        # forward pass
        outputs = model(batch_data)
        loss = loss_fn(outputs, batch_labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # update parameters
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

    # validation loop
    model.eval()
    with tg.no_grad():
        valid_loss = 0
        for batch_data, batchlabels in get_batches(val_images, val_labels, batch_size):
            outputs = model(batch_data)
            valid_loss += loss_fn(outputs, batch_labels).item()
        valid_loss /= len(val_images) / batch_size
    print(f"Validation Loss: {valid_loss}")

# Save the trained model (optional)
model.save("~/Developer/tg-project/models/aeolus.pth")