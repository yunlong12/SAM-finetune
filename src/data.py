import os

import imageio
from torch.utils.data import Dataset, DataLoader

class SegDataset(Dataset):
    def __init__(self, base_dir, mode):
        
        self.data_dir = os.path.join(base_dir, 'Images')
        self.mask_dir = os.path.join(base_dir, 'GroundTruth') if mode == 'train' else None

    def __len__(self):
        return len(os.listdir(self.data_dir))


    def __getitem__(self, idx):

        #idx = 50
        print("idx:{}".format(idx))

        #file = os.listdir(self.data_dir)[idx]
        file = "Image_1.png"
        x = imageio.imread(os.path.join(self.data_dir, file))
        mask_file = file.replace("Image","GT")


        if self.mask_dir:
            mask = imageio.imread(os.path.join(self.mask_dir, mask_file))
        else:
            mask = None

        return x, mask,idx


def trivial_collate(batch):
    return batch[0]

def get_dataloader(base_data_dir, mode):

    dataset = SegDataset(base_data_dir, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        collate_fn=trivial_collate
    )
    return dataloader