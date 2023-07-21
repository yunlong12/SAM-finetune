import torch
import numpy as np
import matplotlib.pyplot as plt


def normalize(tensor):
    # Subtract the minimum of the tensor and divide by the range
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

def convert_tensor(tensor):
    tensor[tensor < 0] = 0  # convert negative values to 0
    tensor[tensor > 0] = 1  # convert positive values to 1
    return tensor

def display_tensor_as_image(preMask,gtMask):
    # First, we squeeze the tensor to remove dimensions of size 1

    preMask = preMask.squeeze()
    gtMask = gtMask.squeeze()

    # Then we detach the tensor from its computational graph, move it to cpu and convert it to a numpy array
    preMask = preMask.detach().cpu().numpy()
    gtMask = gtMask.detach().cpu().numpy()

    # Use imshow to display the tensor as an image
    plt.imshow(preMask, cmap='Reds', alpha=1.0)
    plt.imshow(gtMask, cmap='Blues', alpha=0.5)

    # Specify ticks on both the x and y axes at intervals of 50
    plt.xticks(np.arange(0, preMask.shape[1], 50))
    plt.yticks(np.arange(0, preMask.shape[0], 50))

    # Add a grid
    plt.grid(True)

    plt.show()

loadedTensors = torch.load("tensors.pth")
gt_mask_tensor = loadedTensors["gt_mask_tensor"]
binary_mask = loadedTensors["binary_mask"]
display_tensor_as_image(binary_mask,gt_mask_tensor)
print("hi")