import src.training.train_config as cfg
from src.extract_patches import extract_patches
from src.generate_feature_cubes import generate_feature_cubes
from src.training.train_slide_classifier import train_classifier, test_classifier


def main():
    print(f"Extracting patches using csv file: {cfg.slide_distribution_path}")
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
    print("\nSaving feature cubes...")
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
    print("\nTraining model...")
    _model = train_classifier(
        diagnosis=cfg.diagnosis,
        model_path=cfg.slide_classifier_saving_path,
        data_path=cfg.slide_classifier_data_path,
        batch_size=cfg.slide_classifier_batch_size,
        num_epochs=cfg.slide_classifier_num_epochs,
        lr=cfg.slide_classifier_lr,
        dropout=cfg.slide_classifier_dropout,
    )
    print("\nTesting best model...")
    test_classifier(
        model=_model,
        diagnosis=cfg.diagnosis,
        data_path=cfg.slide_classifier_data_path,
        batch_size=cfg.slide_classifier_batch_size,
    )


if __name__ == "__main__":
    main()
