import pymysql
import pandas as pd
from datetime import datetime
import os
from os import path
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np

source_db = {
    "host" : "219.252.39.14",
    "user": "root",
    "password": "seegeneai2020",
    "db": "seegene_il",
    "table_1": "recommend_slide",
    "table_2": "recommend_patch"
}

target_db = {
    "host" : "219.252.39.14",
    "user": "root",
    "password": "seegeneai2020",
    "db": "seegene_il",
    "table": "recommend_patch"
}

# RP : Recommend patch / RS : Recommend_slide 
QUERY = "SELECT RP.SLIDE_NAME, RS.SLIDE_PATH, RP.PATCH_NAME, RS.SCAN_DATE, RP.X_LOC, RP.Y_LOC, RP.PATCH_GROUNDTRUTH, RP.ORACLE_SELECTION " \
    "FROM RECOMMEND_PATCH AS RP LEFT JOIN RECOMMEND_SLIDE AS RS ON RP.SLIDE_NAME = RS.SLIDE_NAME " \
    "WHERE RP.SLIDE_NAME = '{slide_name}' AND RP.ANATOMY = 'Test' " \
    "AND (RP.ORACLE_SELECTION = 2 OR RP.ORACLE_SELECTION = 4)" 

# DB를 읽어 와서 각 슬라이드 별로 폴더를 따로 만들어 Zoom in 한 patch 이미지를 각각 저장한다. 
def main():
    connection = connect_to_db(source_db)
    with connection.cursor() as cursor:
        all_patches = []
        slide_name = "2022S 0085886010103"
        query = QUERY.format(slide_name = slide_name)
        result = query_execution(cursor, query)
        all_patches.extend(result)
    connection.close()

    if len(getDupTuple(all_patches)) > 0:
        all_patches = removeDupTuple(all_patches)
    
    # os.curdir : 현재 작업하고 있는 디렉토리 정보 
    root_dir = os.path.abspath(os.curdir)
    for p in all_patches:
        path_ = p[3].strftime("%Y%m%d") + '/Patch/' + str(p[7]) + '/' + p[6] # RS.SCAN_DATE의 년/달/일 + '/Patch/' + str(RP.PATCH_GROUNDTRUTH) + '/' + RP.Y_LOC
        folder_path = os.path.join(root_dir, path_)
        # check path exists
        # if not -> create path
        if not path.exists(folder_path):
            # print(folder_path)
            os.makedirs(folder_path)
        
        
        img_path = os.path.join(folder_path, p[2]) # RP.PATCH_NAME 
        img_path = img_path + ".png"
        # print(img_path)
        # check if image exists
        # if not -> generate image
        if not path.exists(img_path):
            slide_path_ = p[1] + ".mrxs" # mrxs 는 슬라이드 MIRAX 시리즈 또는 MIRAX 호환 현미경 디지털 슬라이드 스캐너로 만든 이미지 파일 형식
            # !! windows and linux have different file access of /, \ 
            slide_path_ext = slide_path_.split('/', 3)
            slide_path = "/" + slide_path_ext[-1]
            
            row = int(int(p[5])/240)
            col = int(int(p[4])/240)
            gen_patch = coord_to_patch(slide_path.replace('\\', '/'), row=row, col=col)
            # save .png
            gen_patch.save(img_path)
            
            # # write the generated path to db
            # write_to_db(img_path)
    
    return

# slide의 주소(slide 주소와 추천하는 patch의 주소)를 받아, 추천 patch를 줌인 하여 불러온다.
def coord_to_patch(
    slide_path: str,
    row: int, 
    col: int, 
    tile_size: int = 256,
    overlap: int = 8, 
    resolution_factor: int = 4,
    limit_bounds: bool = False):
    """    
    RETURN PIL.Image
    """

    slide = open_slide(slide_path)
    # overlap이란 각 patch 별로 가장자리에 추가되는 pixel 
    tile_size = tile_size - (2 * overlap)  # => 이렇게 해서 240이 되는 거구나. 
    
    # deep zoom : 고해상도 이미지에 대해서 zoom in / zoom out을 빠르게 진행할 수 있도록 해줌. 
    tiles = DeepZoomGenerator(
        slide,
        tile_size=tile_size,
        overlap=overlap,
        limit_bounds=limit_bounds) # limit bounds : True 면 꽉 차있는 경우에 대해서만 만드는 것 
    
    max_level = tiles.level_count - 1 
    level = max_level - int(np.log(resolution_factor) / np.log(2))
    
    return tiles.get_tile(level, (col, row)) # col, row 위치에 level 정도의 Zoom in을 진행한다. 
    
def connect_to_db(db_config: dict):
    connection = pymysql.Connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        db=db_config['db'],
        autocommit=True)
    return connection

def query_execution(cursor, query: str):
    cursor.execute(query)
    return cursor.fetchall()

def removeDupTuple(lst):
    return [t for t in (set(tuple(i) for i in lst))]

def getDupTuple(lst):
    return list(set([t for t in lst if lst.count(t) > 1]))


if __name__ == '__main__':
    main()