import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path, PureWindowsPath
from src.utils.utils_tiling import save_patches, delete_light_patches

import src.training.train_config as cfg


def get_mirax_files(_dir: str) -> list:
    return list(Path(_dir).glob('**/*.mrxs'))


def create_mirax_file(_file: str) -> None:
    if not Path(_file).exists():
        Path(_file).touch()


def find_mirax_file(_path: str, _name: str) -> str:
    mirax_list = list(Path(_path).glob(f'**/{_name}'))
    if len(mirax_list) == 0:
        return ''
    return str(mirax_list[0])


def distribute_slides(slide_list: list, train_percent: float = 0.7) -> list:
    n = len(slide_list)
    train_n = int(n * train_percent)
    slide_list = random.sample(slide_list, n)
    distribution_array = ['train'] * train_n
    distribution_array += ['val'] * (n - train_n)
    return list(zip(slide_list, distribution_array))


def extract_patches(
        slide_distribution_path: str,
        tile_dir: str,
        tile_size: int,
        overlap: int,
        resolution_factor: float,
        tile_ext: str,
        use_filter: bool = True,
        limit_bounds: bool = False,
        workers: int = 20) -> None:

    assert Path(slide_distribution_path).exists(), f"{slide_distribution_path} does not exist"

    df = pd.read_csv(slide_distribution_path)

    for row in tqdm(df.itertuples(), total=len(df)):
        if not isinstance(row.path, str):
            continue
        # slide_path = Path(find_mirax_file('annotation_only', row.file))
        _path = row.path
        slide_path = PureWindowsPath(_path) if '\\' in _path else Path(_path)

        if not slide_path.exists():
            print(str(slide_path), "does not exist")
            continue
        slide_file = str(slide_path)
        if not Path(slide_file).exists():
            Path(slide_file).touch()
        slide_name = slide_path.stem
        diagnosis = row.label
        mode = row.mode

        assert diagnosis in ('M', 'D', 'N', 'U'), \
            f"{str(slide_path)} do not follow mode/diagnosis/file"
        assert mode in ('train', 'val', 'test'), \
            f"{str(slide_path)} do not follow mode/diagnosis/file"

        tile_path = f"{tile_dir}/{mode}/{diagnosis}/{slide_name}"

        Path(tile_path).mkdir(exist_ok=True, parents=True)

        # overlap = 0 if mode != 'train' else int(tile_size * overlap_factor)
        # overlap = 0 if mode != 'train' else 8

        save_patches(
            slide_file=slide_file,
            output_path=tile_path,
            resolution_factor=resolution_factor,
            tile_size=tile_size,
            overlap=overlap,
            ext=tile_ext,
            use_filter=use_filter,
            limit_bounds=limit_bounds,
            workers=workers
        )

    delete_light_patches(_dir=tile_dir, ext=tile_ext)

    return


if __name__ == "__main__":
    extract_patches(
        slide_distribution_path=cfg.slide_distribution_path,
        tile_dir=cfg.tile_dir,
        tile_size=cfg.tile_size,
        overlap=cfg.overlap,
        resolution_factor=cfg.resolution_factor,
        tile_ext=cfg.tile_ext,
        limit_bounds=cfg.limit_bounds,
        workers=cfg.tiler_workers
    )
