import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import threshold, normalize
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry
import matplotlib.pyplot as plt
import random
from scipy.ndimage import convolve

def get_random_point_in_mask(image):
    kernel = np.ones((21, 21))
    smaller_mask = convolve(image, kernel, mode='constant', cval=0) == kernel.sum()

    # Get indices where smaller_mask is 1
    indices = np.where(smaller_mask == 1)
    indices = np.array(indices).T

    # Select a random index
    if len(indices) == 0:
        return None
    else:
        return tuple(indices[np.random.choice(len(indices))])

class SAMWrapper(nn.Module):
    def __init__(self, ckpt_path, device, from_scratch=False, avg_box=None):
        super().__init__()
        self.device = device
        self.avg_bbox = avg_box

        self.sam_model = sam_model_registry['vit_b'](checkpoint=ckpt_path)
        if from_scratch:
            for layer in self.sam_model.mask_decoder.output_hypernetworks_mlps.children():
                for cc in layer.children():
                    for c in cc.children():
                        try:
                            c.reset_parameters()
                        except:
                            print(f'cannot reset parameters: {c}')

        self.transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)


    def resize_bbox(self, target_size):
        x_scale = target_size[1] / self.avg_bbox[1]
        y_scale = target_size[0] / self.avg_bbox[0]

        self.avg_bbox[[2, 4]] *= y_scale
        self.avg_bbox[[3, 5]] *= x_scale
        self.avg_bbox[:2] = target_size


    def forward(self, X, gt_mask):

        # preprocessing
        original_size = X.shape[:2]
        X = self.transform.apply_image(X)
        X = torch.as_tensor(X, device=self.device)
        X = X.permute(2, 0, 1).contiguous()[None, ...]
        input_size = tuple(X.shape[-2:])
        X = self.sam_model.preprocess(X)

        if gt_mask is not None:
            gt_mask_tensor = torch.from_numpy(gt_mask).float()/ 255.0

            #y,x = torch.where(gt_mask_tensor == 1)
            # bbox1 = np.array([[x.min(), y.min(), x.max(), y.max()]])
            # bbox = self.transform.apply_boxes(bbox1, original_size)
            # bbox_tensor = torch.as_tensor(bbox, dtype=torch.float, device=self.device)


            # points
            # points_per_side_width = 5
            # points_per_side_height = 5
            # left_top_x = 100
            # left_top_y = 100
            # right_bottom_x = 600
            # right_bottom_y = 600
            # image_width = 1280
            # image_height = 720

            # Checks if there is any available GPU
            if torch.cuda.is_available():
                calculateDevice = torch.device('cuda')
            else:
                print('CUDA is not available. Using CPU instead.')
                calculateDevice = torch.device('cpu')

            # points_width = np.linspace(left_top_x, right_bottom_x, points_per_side_width)
            # points_height = np.linspace(left_top_y, right_bottom_y, points_per_side_height)
            # points_x = np.tile(points_width[None, :], (points_per_side_width, 1))
            # points_y = np.tile(points_height[:, None], (1, points_per_side_height))

            #point_y, point_x = get_random_point_in_mask(gt_mask/255)
            point_x = 498
            point_y = 389
            print("point_x:{}, point_y:{}".format(point_x, point_y))
            points_grid_original = np.array([[point_x,point_y]]) #np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
            points_grid = self.transform.apply_coords(points_grid_original,original_size) #this step is very very important

            points_grid_tensor = torch.as_tensor(points_grid, device=calculateDevice)
            points_grid_tensor = points_grid_tensor[:,None,:]
            point_labels = torch.ones(points_grid_tensor.shape[0], dtype=torch.int, device=calculateDevice)
            point_labels = point_labels[:,None]

            points_with_label = (points_grid_tensor, point_labels)

            gt_mask_tensor = gt_mask_tensor.to(self.device)

        
        # model
        with torch.no_grad():
            image_embedding = self.sam_model.image_encoder(X)
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=points_with_label, boxes=None, masks=None
            )
        
        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        upscaled_masks = self.sam_model.postprocess_masks(
            low_res_masks, input_size, original_size
        )
        #binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

        return gt_mask_tensor, upscaled_masks

