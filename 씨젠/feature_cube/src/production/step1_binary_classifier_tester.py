import os.path
import time
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader

import project_config as cfg


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transform: torchvision.transforms):
        super().__init__()
        self.img_paths = list([str(p) for p in Path(root).glob("**/*.jpg")])
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item]).convert("RGB")
        return self.transform(img)


def get_patch_model(patch_classifier_path: str):
    """
    Load patch-level model
    """
    model = torchvision.models.densenet201()
    model.classifier = torch.nn.Linear(model.classifier.in_features, 3)
    model.load_state_dict(torch.load(patch_classifier_path))

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    model.cuda()
    return model


def predict_label(patch_model, data_path: str):
    """
    Predict the binary label for a whole slide. If any of the patches in the slide is labeled as abnormal,
    whole slide is labeled as abnormal
    """

    total_pred = 'N'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tile_image = Dataset(root=data_path, transform=transform)
    tile_loader = DataLoader(tile_image, batch_size=cfg.step1_batch_size, shuffle=False, num_workers=2)

    patch_model.eval()
    with torch.no_grad():
        for inputs in tile_loader:
            inputs = inputs.cuda()
            outputs = patch_model(inputs)
            outputs = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(outputs, 1)

            confidence_threshold = 0.5

            for confidence, prediction in list(zip(confidences, predictions)):
                if (confidence >= confidence_threshold) and (prediction != 0):
                    return "AB"
    return total_pred


def prediction_main(patch_model, slide_name, connection):

    pred = "None"
    # total_start_time = time.perf_counter()
    data_path = os.path.join(cfg.tile_dir, slide_name)
    pred = predict_label(
        patch_model=patch_model,
        data_path=data_path
    )

    # try:
    #     pred = predict_label(
    #         patch_model=patch_model,
    #         data_path=data_path
    #     )
    # except Exception as ex:
    #     print(ex)
    #     # total_time = time.perf_counter() - total_start_time
    #     print(f"Have to check {slide_name}")
    #     ts = time.time()
    #     data_time_processed = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    #     cur = connection.cursor()
    #     sql = "UPDATE `slides_queue` SET `date_time_processed`=%s , `slide_level_label`=%s, decision_step=%s WHERE `slide_name`=%s AND `anatomy`=%s AND `num_classes`=%s"
    #     cur.execute(sql, (data_time_processed, 'E', 'binary', slide_name, cfg.anatomy, 3))
    #     connection.commit()

    return pred


def main_mixpatch(slide_name: str, connection):

    patch_model = get_patch_model(patch_classifier_path=cfg.patch_classifier_path)
    pred = prediction_main(
        patch_model=patch_model,
        slide_name=slide_name,
        connection=connection
    )

    ts = time.time()
    data_time_processed = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    cur = connection.cursor()
    sql = "UPDATE `slides_queue` SET `date_time_processed`=%s , `slide_level_label`=%s, decision_step=%s WHERE `slide_name`=%s AND `anatomy`=%s AND `num_classes`=%s"
    if pred == 'N':
        cur.execute(sql, (data_time_processed, pred, 'binary', slide_name, cfg.anatomy, 3))
    else:
        cur.execute(sql, (data_time_processed, 'Z', 'binary', slide_name, cfg.anatomy, 3))
    connection.commit()
    return pred
