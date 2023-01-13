import torch
import torchvision
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    """
    Load patch images from patch folder using PIL image
    """

    def __init__(self, root: str, transform: torchvision.transforms, labeled: bool):
        super().__init__()
        self.img_paths = list([str(p) for p in Path(root).glob("**/*.jpg")])
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item]).convert("RGB")
        return self.transform(img), self.img_paths[item]


def load_patch_loader(data_path: str, batch_size: int, labeled: bool = False, shuffle: bool = False):
    # load dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tile_image = Dataset(root=data_path, transform=transform, labeled=labeled)
    return DataLoader(tile_image, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def get_patch_model(patch_classifier_path: str):
    """
    Load patch-level model
    """
    model_state_dict = torch.load(patch_classifier_path)
    is_data_parallel = np.array(["module" in k for k in model_state_dict.keys()]).all()
    model = torchvision.models.densenet201()
    model.classifier = torch.nn.Linear(model.classifier.in_features, 3)
    if is_data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.cuda()
    return model


def get_slide_model(slide_model_path: str):
    """
    Load slide-level model
    """
    model_state_dict = torch.load(slide_model_path)
    is_data_parallel = np.array(["module" in k for k in model_state_dict.keys()]).all()

    model = torchvision.models.densenet201()
    model.classifier = torch.nn.Linear(model.classifier.in_features, 3)

    if is_data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(model_state_dict)

    model.cuda()
    return model
