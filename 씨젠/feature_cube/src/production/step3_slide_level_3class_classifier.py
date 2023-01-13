import os
import time
import torch
import pickle
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

import project_config as cfg
from src.utils.utils_torch import get_slide_model


def load_txt_file(data_dir: str):
    file = f"{data_dir}/temp.txt"   
    assert Path(file).exists(), f"{file} does not exist"
    with open(str(file), 'rb') as f:
        features = pickle.load(f)
    return features


def load_feature_cubes(data_dir: str):
    assert Path(data_dir).exists(), f"{data_dir} does not exist"
    return load_txt_file(data_dir)


def slide_level_prediction(slide_model, cube_path, slide_name):

    # Load feature cube
    data = load_feature_cubes(os.path.join(cube_path, slide_name))

    # Predictions
    slide_model.eval()

    with torch.no_grad():
        inputs = torch.tensor(data).float().cuda()
        outputs = slide_model(inputs)
        outputs = F.softmax(outputs, dim=1) 
        cs, prediction = torch.max(outputs, 1)   

    if prediction.item() == 0:
        return 'D'
    elif prediction.item() == 1:
        return 'M'
    elif prediction.item() == 2:
        return 'N'
    else:
        return 'E'


def main_fc(slide_name, connection):

    slide_model = get_slide_model(slide_model_path=cfg.slide_model_path)
    ts = time.time()
    data_time_processed = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    try:
        # Get prediction
        prediction = slide_level_prediction(
            slide_model=slide_model,
            cube_path=cfg.feature_cube_path,
            slide_name=slide_name
        )
        if connection is not None:
            cur = connection.cursor()
            sql = "UPDATE `slides_queue` SET `date_time_processed`=%s , `slide_level_label`=%s, decision_step=%s WHERE `slide_name`=%s AND `anatomy`=%s AND `num_classes`=%s"
            cur.execute(sql, (data_time_processed, prediction, '3class', slide_name, cfg.anatomy, 3))
            connection.commit()
    except Exception as ex:
        print(f"Have to check {slide_name}")
        prediction = 'E'
        if connection is not None:
            cur = connection.cursor()
            sql = "UPDATE `slides_queue` SET `date_time_processed`=%s , `slide_level_label`=%s, decision_step=%s WHERE `slide_name`=%s AND `anatomy`=%s AND `num_classes`=%s"
            cur.execute(sql, (data_time_processed, prediction, '3class', slide_name, cfg.anatomy, 3))
            connection.commit()

    return prediction
