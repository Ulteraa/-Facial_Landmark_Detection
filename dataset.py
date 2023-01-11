import numpy as np
import pandas as pd
import os

import torch
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import MTCNN
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import config
import matplotlib.pyplot as plt
def extract_images_from_csv(csv, column, save_folder, resize=(96, 96)):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # mtcnn = MTCNN(image_size=96)

    for idx, image in enumerate(csv[column]):
        image = np.array(image.split()).astype(np.uint8)
        image = image.reshape(resize[0], resize[1])
        stacked_img = np.stack((image,) * 3, axis=-1)
        img = Image.fromarray(stacked_img)
        # img_cropped = mtcnn(img, save_path=save_folder+f"img_{idx}.png")
        img.save(save_folder+f"img_{idx}.png")

class Dataset_(Dataset):
    def __init__(self, root, csv_file, train=True, transform=None):
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.root = root
        self.category_names = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y',
                               'left_eye_inner_corner_x', 'left_eye_inner_corner_y', 'left_eye_outer_corner_x',
                               'left_eye_outer_corner_y', 'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
                               'right_eye_outer_corner_x', 'right_eye_outer_corner_y', 'left_eyebrow_inner_end_x',
                               'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
                               'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x',
                               'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y', 'mouth_left_corner_x',
                               'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y',
                               'mouth_center_top_lip_x', 'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x',
                               'mouth_center_bottom_lip_y']
        self.transform = transform
        self.train = train

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.train:
            path = self.root+f"img_{index}.png"
            image = np.array(Image.open(path).convert('RGB'))
            labels = np.array(self.data.iloc[index, :30].tolist())
            labels[np.isnan(labels)] = -1
        else:
            #image = np.array(self.data.iloc[index, 1].split()).astype(np.float32)
            path = self.root + f"img_{index}.png"
            image = np.array(Image.open(path).convert('RGB'))
            labels = np.zeros(30)

        ignore_indices = labels == -1
        labels = labels.reshape(15, 2)

        if self.transform:
            # image = np.repeat(image.reshape(96, 96, 1), 3, 2).astype(np.uint8)
            augmentations = self.transform(image=image, keypoints=labels)
            image = augmentations["image"]
            labels = augmentations["keypoints"]
        else:
            image = torch.Tensor(image)
            image = torch.permute(image, (2,0,1))


        labels = np.array(labels).reshape(-1)
        labels[ignore_indices] = -1

        return image, labels.astype(np.float32)

if __name__ == "__main__":
    ds = Dataset_(root='data/train/', csv_file="training/training.csv", train=True, transform=None)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    # resnet = InceptionResnetV1(pretrained='vggface2').eval()
    for idx, (x, y) in enumerate(loader):
        x=torch.permute(x, (0, 3, 1, 2))

        # print(y[0][0::2].detach().cpu().numpy(), y[0][1::2].detach().cpu().numpy())
        # res = resnet(x)
        # plt.imshow(x.squeeze(0))
        plt.imshow(x[0][1].detach().cpu().numpy(), cmap='gray')
        # plt.plot(y[0][0::2].detach().cpu().numpy(), y[0][1::2].detach().cpu().numpy(), "go")
        plt.show()
# if __name__ == '__main__':
#     # ----test data_extraction
#     csv = pd.read_csv("test/test.csv")
#     extract_images_from_csv(csv, "Image", "data/test/")
#     # ----train_data_extraction
#     csv = pd.read_csv("training/training.csv")
#     extract_images_from_csv(csv, "Image", "data/train/")
