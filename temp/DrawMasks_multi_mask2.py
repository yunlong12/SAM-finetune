import torch
import numpy as np
import matplotlib.pyplot as plt

#torch.save({"gt_mask_tensor":gt_mask_tensor,"binary_mask":binary_mask},"./temp/multi_mask_tensor3.pth")

def normalize(tensor):
    # Subtract the minimum of the tensor and divide by the range
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

def convert_tensor(tensor):
    tensor[tensor < 0] = 0  # convert negative values to 0
    tensor[tensor > 0] = 1  # convert positive values to 1
    return tensor

def display_tensor_as_image(tensor1, tensor2, tensor3):
    # First, we squeeze the tensors to remove dimensions of size 1
    tensor1 = tensor1.squeeze()
    tensor2 = tensor2.squeeze()
    tensor3 = tensor3.squeeze()

    # Then we detach the tensors from their computational graph, move them to cpu and convert them to numpy arrays
    tensor1 = tensor1.detach().cpu().numpy()
    tensor2 = tensor2.detach().cpu().numpy()
    tensor3 = tensor3.detach().cpu().numpy()

    # Create a blank RGB image
    image = np.zeros((tensor1.shape[0], tensor1.shape[1], 3))

    # Assign each tensor to a color channel in the image
    image[:, :, 0] = tensor1  # Red channel
    image[:, :, 1] = tensor2  # Green channel
    image[:, :, 2] = tensor3  # Blue channel

    # Use imshow to display the image
    plt.imshow(image)

    # Specify ticks on both the x and y axes at intervals of 50
    plt.xticks(np.arange(0, tensor1.shape[1], 50))
    plt.yticks(np.arange(0, tensor1.shape[0], 50))

    # Add a grid
    plt.grid(True)

    # Invert the y-axis to make the origin the lower-left corner
    plt.gca().invert_yaxis()

    plt.show()

def display_tensor_as_image_InARow(tensor1, tensor2, tensor3):
    # First, we squeeze the tensors to remove dimensions of size 1
    tensor1 = tensor1.squeeze()
    tensor2 = tensor2.squeeze()
    tensor3 = tensor3.squeeze()

    # Then we detach the tensors from their computational graph, move them to cpu and convert them to numpy arrays
    tensor1 = tensor1.detach().cpu().numpy()
    tensor2 = tensor2.detach().cpu().numpy()
    tensor3 = tensor3.detach().cpu().numpy()

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))  # creating a grid of 3 plots

    # plot tensor1
    axs[0].imshow(tensor1, cmap='Reds')
    axs[0].set_title('Tensor 1')
    axs[0].grid(True)
    axs[0].invert_yaxis()  # flip the y-axis

    # plot tensor2
    axs[1].imshow(tensor2, cmap='Greens')
    axs[1].set_title('Tensor 2')
    axs[1].grid(True)
    axs[1].invert_yaxis()  # flip the y-axis

    # plot tensor3
    axs[2].imshow(tensor3, cmap='Blues')
    axs[2].set_title('Tensor 3')
    axs[2].grid(True)
    axs[2].invert_yaxis()  # flip the y-axis

    plt.show()

loadedTensors = torch.load("multi_mask_tensors2.pth")
multi_binary_mask = loadedTensors["multi_binary_mask"]
tensor1,tensor2,tensor3 = torch.split(multi_binary_mask,split_size_or_sections=1,dim=1)
display_tensor_as_image(tensor1,tensor2,tensor3)
display_tensor_as_image_InARow(tensor1,tensor2,tensor3)
print("hi")