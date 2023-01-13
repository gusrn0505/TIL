import os
import shutil
import pymysql
import pandas as pd
from pathlib import Path

import project_config as cfg
import src.production.step1_binary_classifier_tester as step1
import src.production.step2_patch_level_3class_classifier as step2
import src.production.step3_slide_level_3class_classifier as step3

from src.utils.utils_tiling import save_patches, delete_light_patches, count_saved_patches


def slides_to_be_processed_count(connection):
    select_cursor = connection.cursor()
    # slide_label이 아직 부여되지 않은 것만 불러오기. 
    select_cursor.execute("SELECT slide_name FROM slides_queue WHERE slide_level_label=%s AND anatomy=%s AND num_classes=%s", ('', cfg.anatomy, 3))
    total_slides = select_cursor.rowcount
    select_cursor.close()
    print(f"Slides to be processed: {total_slides}")
    return total_slides


def prediction(connection):
    # Load info of slide that has no prediction...
    cur = connection.cursor()
    sql = "SELECT slide_name, slide_path FROM slides_queue WHERE slide_level_label = %s AND anatomy =%s AND num_classes = %s order by date_time_added"
    cur.execute(sql, ('', cfg.anatomy, 3))

    # Conduct the framework for each slide
    rows = cur.fetchall()
    for idx, row in enumerate(rows):
        if not Path(row['slide_path']).exists():
            print(f"{row['slide_path']} does not exist")
            continue
        slide_path = f"{row['slide_path']}.mrxs"
        # Sometimes the mrxs file does not exist even though the folder containing the .dat files does
        # In this case, manually create the .mrxs file
        if not Path(slide_path).exists():
            Path(slide_path).touch()
        slide_name = row['slide_name']

        print(f"__ Processing {slide_name} ")

        # Get the tiles
        tile_dir = os.path.join(cfg.tile_dir, slide_name)
        Path(tile_dir).mkdir(exist_ok=True, parents=True)
        print(f"___ Saving patches at {tile_dir}")
        save_patches(
            slide_file=slide_path,
            output_path=tile_dir,
            resolution_factor=cfg.resolution_factor,
            tile_size=cfg.tile_size,
            overlap=cfg.overlap,
            limit_bounds=cfg.limit_bounds,
            ext=cfg.tile_ext,
            use_filter=cfg.use_filter,
        )
        delete_light_patches(tile_dir, ext=cfg.tile_ext)

        # 임시값. 
        slide_prediction = "AB"

        # Count the number of saved patches, if it is 0, then ignore the slide
        saved_patches = count_saved_patches(tile_dir, ext=cfg.tile_ext)
        if saved_patches == 0:
            shutil.rmtree(tile_dir)
            slide_prediction = 'E'
 
        # Step 1: Conduct binary classification
        # slide_prediction = step1.main_mixpatch(slide_name=slide_name, connection=connection)

        # print("Start 3-class classification")
        if slide_prediction == "AB":
            # Step 2: Feature cube extraction
            print("___ Extracting feature cube")
            slide_prediction = step2.main_fc(
                slide_path=slide_path,
                slide_name=slide_name, 
                connection=connection)

            if slide_prediction != 'E':
                print("___ Slide-level prediction")
                slide_prediction = step3.main_fc(
                    slide_name=slide_name, 
                    connection=connection)

        if cfg.delete_tile_dir:
            shutil.rmtree(tile_dir)

        if slide_prediction == 'E':
            print(f"{slide_path} has issue!")

    total_slides = slides_to_be_processed_count(connection)
    if total_slides != 0:
        print(f"{total_slides} new slides added in Queue. Starting processing recently added slides.")
        prediction(connection)

    return


def main():
    connection = pymysql.connect(
        host=cfg.mariadb["host"],
        user=cfg.mariadb["user"],
        password=cfg.mariadb["password"],
        db=cfg.mariadb["db"],
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    with connection.cursor() as cursor:
        sql = "INSERT INTO `slides_queue` (`slide_name`, `slide_path`, `anatomy`, `num_classes`) VALUES (%s, %s, %s, %s)"
        
        # 여기서 use_csv_file이 어떤 맥락으로 나온거지? csv 파일을 따로 저장하는 건가? 
        
        # slide label prediction 결과와 주소가 정리된 건가? 
        if cfg.use_csv_file:
            # If the list of slides is stored in a csv file then read the csv files and take those
            # slides labeled as "test". It is possible to remove the second line so you can take all
            # the paths in the csv file if needed.

            # slide_distribution_path = f"./{anatomy.lower()}_slide_distribution.csv"
            slide_csv = pd.read_csv(cfg.slide_distribution_path)
            # 흠.. 내가 찾은 csv 파일과는 또 다른 건가 보다. mode란 항목이 없네. 
            slide_csv = slide_csv.loc[slide_csv['mode'] == 'test']
            slide_paths = slide_csv.path.tolist()
        else:
            # If you want to read the slides directly from a directory, then the slides should be directly under the
            # directory and not under a subfolder.
            # The WSI are listed in form of directories, the mrxs files are ignored.
            slide_paths = [str(p) for p in Path(cfg.slide_path).iterdir() if p.is_dir()]
        for slide_path in slide_paths:
            slide_name = Path(slide_path).stem
            if slide_path[-5:] == ".mrxs":
                slide_path = slide_path[:-5]
            cursor.execute(sql, (slide_name, slide_path, cfg.anatomy, 3))

    connection.commit()
    prediction(connection)
    connection.close()


if __name__ == "__main__":
    main()


