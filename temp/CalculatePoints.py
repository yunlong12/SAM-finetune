import torch
import numpy as np
import matplotlib.pyplot as plt

# points
points_per_side_width = 5
points_per_side_height = 5
left_top_x =100
left_top_y=100
right_bottom_x = 600
right_bottom_y = 600
image_width = 1280
image_height = 720
calculateDevice = torch.device('cpu')

points_width = np.linspace(left_top_x, right_bottom_x, points_per_side_width)
points_height = np.linspace(left_top_y, right_bottom_y, points_per_side_height)
points_x = np.tile(points_width[None, :], (points_per_side_width, 1))
points_y = np.tile(points_height[:, None], (1, points_per_side_height))
points_grid = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)

points_grid_tensor = torch.as_tensor(points_grid, device=calculateDevice)
point_labels = torch.ones(points_grid_tensor.shape[0], dtype=torch.int, device=calculateDevice)

points_with_label = (points_grid_tensor, point_labels)

print(points_grid)
