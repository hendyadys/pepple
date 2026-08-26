import os, cv2, json, random, math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

from sys import platform
if platform == "linux" or platform == "linux2":
    BASE_FOLDER = '/data/yue/pepple'
elif platform == "win32":
    BASE_FOLDER = 'z:/yue/pepple'
DATA_FOLDER = os.path.join(BASE_FOLDER, 'classification', 'data')
DATA_FOLDER = os.path.join(BASE_FOLDER, 'classification', 'data', 'Images for AI analysis')
DATA_FOLDER = os.path.join(BASE_FOLDER, 'classification', 'data', '2021.07.07 New TIFFs')
NUM_CLASSES = 6


class PeppleDataset_mixup(Dataset):
    def __init__(self, img_labels, transform=None):    # default to training masks
        """
        Args:            
            img_names (string): names of all images to load
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.img_labels = img_labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx, visualise=False):
        img_label = self.img_labels[idx]

        img, img_score = get_img_mask_simple(img_label)

        # pytorch unlike keras takes integer class label.
        # see examples-master/MNIST or https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        # target = make_one_hot_from_class_val([int(target)])   # keras style

        H, W, _ = img.shape
        lambda_var = random.random()
        jdx = random.sample(range(len(self.img_labels)), 1)[0]
        img_label_j = self.img_labels[jdx]
        img_j, img_score_j = get_img_mask_simple(img_label_j)
        img_mixed = img * lambda_var + img_j * (1- lambda_var)
        img_pil = Image.fromarray(img_mixed.astype(np.uint8))
        img_score_out = np.zeros(NUM_CLASSES).astype(np.float32)

        img_score_out[int(img_score)] = lambda_var
        img_score_out[int(img_score_j)] = (1- lambda_var)

        if self.transform:
            trans_img = self.transform(img_pil)

            if visualise:
                plt.figure(1)
                plt.imshow(np.asarray(img_pil).astype(np.uint8))
                plt.title('original img')

            return trans_img, img_score_out
        else:
            return img_pil, img_score_out


# get 1 img
def get_img_mask_simple(img_label, visualise=False):
    img_score, img_path = img_label
    if platform == "win32":
        img_path = img_path.replace('/data/yue/pepple/', 'z:/yue/pepple/')
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path)   # color
    # img_pil = Image.fromarray(img.astype(np.uint8))

    if visualise:
        plt.figure(1)
        plt.imshow(img)
    return img, img_score


def get_imgs_for_split(region='AC', nrow=512, ncol=512):
    combined_img_folder = os.path.join(DATA_FOLDER, '{}_imgs_r{}_c{}'.format(region, nrow, ncol))

    splits = ['training', 'validation', 'test']
    train_img_labels = []
    split_file = os.path.join(combined_img_folder, 'training_img_labels.csv')
    # split_file = os.path.join(combined_img_folder, 'training_img_labels_windows.csv')   # for testing purposes
    with open(split_file, 'r') as fin:
        lines = fin.readlines()
        for l in lines:
            l_toks = l.rstrip().split(',')
            score, img_path = l_toks
            score = float(score)
            if score==0.5:
                score = 5   # make it at end
            train_img_labels.append([score, img_path])
    fin.close()
    print(combined_img_folder, 'train', len(train_img_labels))

    valid_img_labels = []
    split_file = os.path.join(combined_img_folder, 'validation_img_labels.csv')
    with open(split_file, 'r') as fin:
        lines = fin.readlines()
        for l in lines:
            l_toks = l.rstrip().split(',')
            score, img_path = l_toks
            score = float(score)
            if score == 0.5:
                score = 5  # make it at end
            valid_img_labels.append([score, img_path])
    fin.close()
    print(combined_img_folder, 'valid', len(valid_img_labels))

    test_img_labels = []
    split_file = os.path.join(combined_img_folder, 'test_img_labels.csv')
    with open(split_file, 'r') as fin:
        lines = fin.readlines()
        for l in lines:
            l_toks = l.rstrip().split(',')
            score, img_path = l_toks
            score = float(score)
            if score == 0.5:
                score = 5  # make it at end
            test_img_labels.append([score, img_path])
    fin.close()
    print(combined_img_folder, 'test', len(test_img_labels))

    return train_img_labels, valid_img_labels, test_img_labels


if __name__ == "__main__":
    train_img_labels, valid_img_labels, test_img_labels = get_imgs_for_split(region='AC')

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512), Image.NEAREST),
        transforms.RandomAffine(degrees=(0, 360), scale=(0.9, 1.1), shear=(.9, 1.1)),
        transforms.RandomHorizontalFlip(),  # not done in graham
        transforms.RandomVerticalFlip(),  # not done in graham
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    k_ds = PeppleDataset_mixup(train_img_labels, transform=transform)
    for idx in range(10):
        trans_img, trans_reg, target, img_name = k_ds.__getitem__(0)