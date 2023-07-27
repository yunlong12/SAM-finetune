import torch
import numpy as np
import matplotlib.pyplot as plt
import os

#torch.save({"gt_mask_tensor":gt_mask_tensor,"binary_mask":binary_mask},"./temp/multi_mask_tensors3.pth")

#torch.save({"gt_mask_tensor":gt_mask_tensor,"binary_mask":binary_mask},"./temp/multi_mask_tensor_onlyOneMask_Point_250_350.pth")
#torch.save({"pointsCoordinates":torch.as_tensor(points_grid_original, device=calculateDevice)},"./temp/points_250_350.pth")

def normalize(tensor):
    # Subtract the minimum of the tensor and divide by the range
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

def convert_tensor(tensor):
    tensor[tensor < 0] = 0  # convert negative values to 0
    tensor[tensor > 0] = 1  # convert positive values to 1
    return tensor


def save_tensor_as_image(tensor, points, directory="./output_onePoint_250_350"):
    # Ensure the output directory exists
    os.makedirs(directory, exist_ok=True)

    # First, we squeeze the tensors to remove dimensions of size 1
    tensor = tensor.squeeze(0)
    #points = points.squeeze(0)

    # Then we detach the tensors from their computational graph, move them to cpu and convert them to numpy arrays
    tensor = tensor.detach().cpu().numpy()
    points = points.detach().cpu().numpy()

    for i in range(tensor.shape[0]):
        plt.imshow(tensor[i], cmap='gray')

        # Draw the point
        plt.scatter(points[i][0], points[i][1], color='red')  # Inverted indices due to y-x coordinates


        plt.grid(True)
        plt.title(f'Slice {i + 1}')
        plt.savefig(f'{directory}/image_{i + 1}.png')
        plt.close()


loadedTensors = torch.load("multi_mask_tensor_onlyOneMask_Point_250_350.pth")
loadedTensors2 = torch.load("points_250_350.pth")
multi_binary_mask = loadedTensors["binary_mask"]
points = loadedTensors2["pointsCoordinates"]
save_tensor_as_image(multi_binary_mask,points)
print("hi")