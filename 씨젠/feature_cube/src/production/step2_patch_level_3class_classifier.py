import os
import time
import numpy as np
from datetime import datetime
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator

import project_config as cfg
from src.utils.utils_torch import get_patch_model
from src.utils.utils_feature_cube import build_feature_cube, save_feature_cubes


def main_fc(slide_path: str, slide_name: str, connection):
    pred = 'E'
    patch_model = get_patch_model(patch_classifier_path=cfg.patch_classifier_path)

    try:
        slide = open_slide(slide_path)
        tiles = DeepZoomGenerator(
            slide,
            tile_size=cfg.tile_size - (cfg.overlap * 2),
            overlap=cfg.overlap,\
            limit_bounds=cfg.limit_bounds
        )
        max_level = tiles.level_count - 1
        level = max_level - int(np.log(cfg.resolution_factor) / np.log(2))
        slide_width, slide_height = tiles.level_dimensions[level]

        # Save patch-level predictions into DB
        data_path = os.path.join(cfg.tile_dir, slide_name)
        patch_predictions, slide_features = build_feature_cube(
            patch_model=patch_model,
            data_path=data_path,
            lsize=cfg.lsize,
            csize=cfg.csize,
            overlap=cfg.overlap,
            batch_size=cfg.step2_batch_size,
            tile_size=cfg.tile_size
        )

        save_feature_cubes(
            slide_features=slide_features,
            slide_name=slide_name,
            feature_cube_path=cfg.feature_cube_path
        )

        if connection is not None:
            cur = connection.cursor()
        for idx, prediction in enumerate(patch_predictions):
            if prediction[2] == 0:
                pred = 'D'
            if prediction[2] == 1:
                pred = 'M'
            elif prediction[2] == 2:
                pred = 'N'
            ts = time.time()
            timestamp = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            if connection is not None:
                sql = "INSERT INTO ai_predictions_copy (model_version, slide_name,anatomy, x_scale, y_scale, x_loc, y_loc, patch_label, date_time, num_classes) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                cur.execute(sql, ('stomach_ai_framework_3', slide_name, cfg.anatomy, str(slide_width), str(slide_height),
                                  str(prediction[0]), str(prediction[1]), pred, timestamp, 3))
    except Exception as ex:
        print(f"Have to check {slide_name}")
        pred = 'E'
        if connection is not None:
            ts = time.time()
            data_time_processed = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            cur = connection.cursor()
            sql = "UPDATE `slides_queue` SET `date_time_processed`=%s , `slide_level_label`=%s, decision_step=%s WHERE `slide_name`=%s AND `anatomy`=%s AND `num_classes`=%s"
            cur.execute(sql, (data_time_processed, pred, '2Class', slide_name, cfg.anatomy, 3))
            connection.commit()

    return pred
