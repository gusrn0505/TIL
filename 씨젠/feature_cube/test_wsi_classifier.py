import os
import time
import shutil
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score

import project_config as cfg
import src.production.step1_binary_classifier_tester as step1
import src.production.step2_patch_level_3class_classifier as step2
import src.production.step3_slide_level_3class_classifier as step3

from src.utils.utils_tiling import save_patches, delete_light_patches


def main():

    true_values = []
    pred_values = []
    class_to_idx = {k: v for v, k in enumerate(sorted(['D', 'M', 'N']))}

    slide_csv = pd.read_csv(cfg.slide_distribution_path)
    slide_csv = slide_csv.loc[slide_csv['mode'] == 'test']
    rows = slide_csv.path.tolist()
    labels = slide_csv.label.tolist()

    # Conduct the framework for each slide
    corrects, total = 0, 0
    for idx, row in enumerate(rows):
        slide_path = row
        if not Path(slide_path).exists():
            Path(slide_path).touch()

        slide_name = Path(row).stem

        ground_truth = labels[idx]

        ts = time.perf_counter()

        # Get the tiles
        tile_dir = os.path.join(cfg.tile_dir, slide_name)
        Path(tile_dir).mkdir(exist_ok=True, parents=True)
        save_patches(
            slide_file=slide_path,
            output_path=tile_dir,
            resolution_factor=cfg.resolution_factor,
            tile_size=cfg.tile_size,
            overlap=cfg.overlap,
            limit_bounds=cfg.limit_bounds,
            ext=cfg.tile_ext,
            use_filter=cfg.use_filter
        )
        delete_light_patches(tile_dir)

        # Step 2: Feature cube extraction
        slide_prediction = step2.main_fc(
            slide_path=slide_path,
            slide_name=slide_name,
            connection=None)

        if slide_prediction != 'E':
            slide_prediction = step3.main_fc(
                slide_name=slide_name,
                connection=None)

        total_time = time.perf_counter() - ts
        print(f"{idx + 1}/{len(rows)} | {slide_name} - GT: {ground_truth}, Pred: {slide_prediction} | {total_time / 60:.2f}mins")

        if slide_prediction != 'E':
            corrects += int(ground_truth == slide_prediction)
            total += 1
            true_values.append(class_to_idx[ground_truth])
            pred_values.append(class_to_idx[slide_prediction])

        if cfg.delete_tile_dir:
            shutil.rmtree(tile_dir)

        if slide_prediction == 'E':
            print(f"{slide_path} has issue!")

    print(f"Accuracy: {accuracy_score(true_values, pred_values):.4f}")
    print(confusion_matrix(true_values, pred_values))
    return


if __name__ == "__main__":
    main()


