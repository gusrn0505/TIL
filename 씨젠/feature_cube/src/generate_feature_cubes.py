import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from src.utils.utils_tiling import save_patches, delete_light_patches
from src.utils.utils_torch import get_patch_model
from src.utils.utils_feature_cube import base_feature_cube, add_pred_to_base, predict_patch_label

import src.training.train_config as cfg

# txt 파일에다가 각 patch 차원에 대해서 prediction 값을 부여한다! 
def extract_slide_feature(
        patch_model,
        slide_path: str,
        tile_dir: str,
        condition: str,
        subset: str,
        saving_path: str,
        lsize: int,
        csize: int,
        batch_size: int,
        resolution_factor: float,
        tile_size: int,
        overlap: int,
        already_patched: bool) -> None:
    """
    Extract features from the slide to make the cube
    """

    # Slide to patches
    if '\\' in slide_path:
        slide_path = '/'.join(slide_path.split('\\'))
    slide_name = Path(slide_path).stem

    tile_path = f"{tile_dir}/{subset}/{condition}/{slide_name}"

    # 이미 patch로 나눠지지 않았다면 save_patch 함수를 통해서 jpg 형태로 저장하기. 
    if not already_patched:
        Path(tile_path).mkdir(exist_ok=True, parents=True)

        save_patches(
            slide_file=slide_path,
            output_path=tile_path,
            resolution_factor=resolution_factor,
            tile_size=tile_size,
            overlap=overlap,
            ext="jpg",
            use_filter=True,
            workers=8
        )
        delete_light_patches(_dir=tile_path, ext="jpg")

    # Getting predictions
    if not Path(tile_path).exists():
        print(f"{tile_path} does not exist")
        return

    patch_level_pred = predict_patch_label(data_path=tile_path, patch_model=patch_model, batch_size=batch_size)

    # Saving feature cubes
    Path(f"{saving_path}/{subset}/{condition}").mkdir(exist_ok=True, parents=True)
    # base_feature_Cube : 데이터 형상 만들기. 내부 값은 전부 0!
    bf = base_feature_cube(lsize, csize)
    slide_feature, patch_pred = add_pred_to_base(
        base_feature=bf,
        predict=patch_level_pred,
        overlap=overlap,
        tile_size=tile_size)

    # 여기서 txt 파일로 저장하는 구만!
    # pickle은 대상의 상태를 bynary로 변환시켜준다 (wb) 이러니까 그냥 메모장을 열었을 때 알아보기 어려웠구나 
    # "{saving_path}/{subset}/{condition}/{slide_name}.txt" 에 대해서 slide_feature.tolist() 정보를 저장해라 라는 거네. 
    # 여기서 patch_pred 정보를 넣지 않아도 되나? 이게 핵심아닌가? 
    # 아 slide_feature 에 대해서 add_pred_to_base 함수에서 predict 값을 집어넣는구나 
    # base_feature[0][i][int(col)][int(row)] = [pred[i]] . 여기서 i는 NDM 의 값! 
    with open(f"{saving_path}/{subset}/{condition}/{slide_name}.txt", "wb") as f:
        pickle.dump(slide_feature.tolist(), f)

    return


def generate_feature_cubes(
        slide_distribution_path: str,
        tile_dir: str,
        patch_classifier_path: str,
        feature_cube_path: str,
        lsize: int,
        csize: int,
        batch_size: int,
        tile_size: int,
        resolution_factor: float,
        overlap: int,
        already_patched: bool):
    """
    Generate feature cube per slide
    """
    # Loading model
    model = get_patch_model(patch_classifier_path=patch_classifier_path)

    # Generate cubes per slide
    slide_distribution = pd.read_csv(slide_distribution_path)

    for row in tqdm(slide_distribution.itertuples(), total=len(slide_distribution)):
        # overlap = 0 if row.mode != 'train' else int(tile_size * overlap_factor)
        extract_slide_feature(
            patch_model=model,
            slide_path=row.path,
            tile_dir=tile_dir,
            condition=row.label,
            subset=row.mode,
            saving_path=feature_cube_path,
            lsize=lsize,
            csize=csize,
            batch_size=batch_size,
            resolution_factor=resolution_factor,
            tile_size=tile_size,
            overlap=overlap,
            already_patched=already_patched,
        )

    return


if __name__ == "__main__":
    generate_feature_cubes(
        slide_distribution_path=cfg.slide_distribution_path,
        patch_classifier_path=cfg.feature_cube_patch_classifier_path,
        tile_dir=cfg.tile_dir,
        feature_cube_path=cfg.feature_cube_path,
        lsize=cfg.lsize,
        csize=cfg.csize,
        batch_size=cfg.feature_cube_batch_size,
        tile_size=cfg.tile_size,
        resolution_factor=cfg.resolution_factor,
        overlap=cfg.overlap,
        already_patched=True,
    )
