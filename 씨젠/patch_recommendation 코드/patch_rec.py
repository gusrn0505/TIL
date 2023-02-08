import pymysql
import pandas as pd

source_db = {
    "host" : "219.252.39.14",
    "user": "root",
    "password": "seegeneai2020",
    "db": "seegene_il",
    "table_1": "ai_predictions",
    "table_2": "recommend_slide",
}
# recommend_slide - 무진님이 보내주신 PPT 파일에 있음 
# recomment_patch - 무진님이 보내주신 PPT 파일에 있음. 


target_db = {
    "host" : "219.252.39.14",
    "user": "root",
    "password": "seegeneai2020",
    "db": "seegene_il",
    "table": "recommend_patch"
}

QUERY = "SELECT AP.SLIDE_NAME, AP.PATCH_LABEL, AP.LABEL_P, AP.X_LOC, AP.Y_LOC " \
    "FROM AI_PREDICTIONS AS AP " \
    "WHERE AP.SLIDE_NAME = '{slide_name}' " \
    "ORDER BY AP.X_LOC, AP.Y_LOC"

# ai_prediction DB에 있는 값들을 기반으로 신뢰도가 0.85보다 낮은 patch와 그 주변 patch들을 sampling 한다. 
# Sampling 한 patch들을 DB의 Recommend_patch table 에 값을 넣어라 
def main():
    connection = connect_to_db(source_db)
    with connection.cursor() as cursor:
        all_patches = []
        slide_name = "2022S 0085886010103"
        # slide_name 에 해당하는 slide 불러오기. 
        query = QUERY.format(slide_name = slide_name)
        result = query_execution(cursor, query)
        all_patches.extend(result)
    connection.close()


    # getDupTuple : 1개보다 많은 항목들에 대해서 Set으로 중복을 제거한 후 list로 반환.
    # 그런데 이러면 1개만 있는 경우는 포함이 안되지 않나? 왜 1보다 많은 경우에 대해서만 불러냈을까?
    # 아하 이건 Patch에 대한 classification 결과구나. 우리는 군집 분석을 할 것이니까 1개 이상이 속한 것만 불러오는 게 맞다. 

    if len(getDupTuple(all_patches)) > 0:
        all_patches = removeDupTuple(all_patches) #all_patch 의 각 항목에 대해서 중복 제거 
    
    threshold = 0.85
    # 신뢰도가 0.85 보다 낮은 값들을 학습 대상으로 추천함. 
    rec_center_patch = [patch for patch in all_patches if patch[2] < threshold] # patch[2] = AP.LABEL_P - 신뢰도를 의미

    all_rec_patches = []
    for cp in rec_center_patch:
        x = cp[3] # AP.X_LOC
        y = cp[4] # AP.Y_LOC
        #surrounding patches + center patches. 240 간격 안에 있는 patch들도 포함시키기. 
        all_rec_patches.extend(patch for patch in all_patches if (abs(int(patch[3]) - int(x))/240 <= 1) and 
                                                                 (abs(int(patch[4]) - int(y))/240 <= 1) and
                                                                 (patch not in all_rec_patches))

    # ANATOMY, X_SCALE, Y_SCALE might be removes later
    target = connect_to_db(target_db)
    # 조건을 만족하는 DB에 대해서 RECOMMEND_PATCH 에 값을 추가해라. 여기서 recommend patch에 값을 추가하는 구나. 
    query = "INSERT INTO RECOMMEND_PATCH (SLIDE_NAME, PATCH_NAME, PATCH_PREDICTION, PATCH_SCORE, X_LOC, Y_LOC, PATCH_GEN_PATH, PATCH_GROUNDTRUTH, ORACLE_SELECTION, ANATOMY, X_SCALE, Y_SCALE) " \
        "VALUES ('{slide_name}', '{patch_name}', '{patch_prediction}', '{patch_score}', '{x_loc}', '{y_loc}', '{patch_gen_path}', '{patch_groundtruth}', '{oracle_selection}', '{anatomy}', '{x_scale}', '{y_scale}')"

    with target.cursor() as cursor:
        for patch in all_patches:
            patch_name = patch[0] + "_" + patch[3] + "_" + patch[4] # AP.SLIDE_NAME _ AP.X_LOC _ AP.Y_LOC
            oracle_selection = 1 if patch in all_rec_patches else 0 # 추천하는 것들은 1의 값 부여, 그 외는 0의 값 부여. 
            query_ = query.format(
                slide_name = patch[0],
                patch_name = patch_name,
                patch_prediction = patch[1],
                patch_score = patch[2],
                x_loc = patch[3],
                y_loc = patch[4],
                patch_gen_path = "",
                patch_groundtruth = "",
                oracle_selection = oracle_selection,
                anatomy = "Test",
                x_scale = 22509,
                y_scale = 53169
            )
            # print(query_)
            query_execution(cursor, query_)
    target.close()
    return 

def rec_patch(slides):

    for slide in slides:
        slide_name_ = slide[1]
        connection = connect_to_db(source_db)
        with connection.cursor() as cursor:
            all_patches = []
            slide_name = slide_name_
            query = QUERY.format(slide_name = slide_name)
            result = query_execution(cursor, query)
            all_patches.extend(result)
        connection.close()

        if len(getDupTuple(all_patches)) > 0:  
            all_patches = removeDupTuple(all_patches) 
        
        threshold = 0.85
        rec_center_patch = [patch for patch in all_patches if patch[2] < threshold] # patch[2]의 정보가 뭘까? Loss diff을 적용했다 가정하면 Loss 값이 되겠지?

        all_rec_patches = []
        for cp in rec_center_patch: 
            x = cp[3]
            y = cp[4]
            #surrounding patches + center patches
            all_rec_patches.extend(patch for patch in all_patches if (abs(int(patch[3]) - int(x))/240 <= 1) and 
                                                                    (abs(int(patch[4]) - int(y))/240 <= 1) and
                                                                    (patch not in all_rec_patches))
        
        print(f"slide name: {slide_name_}, #patches = {len(all_patches)}, #rec_patches:{len(all_rec_patches)}")
        
        # # ANATOMY, X_SCALE, Y_SCALE might be removes later
        # target = connect_to_db(target_db)
        # query = "INSERT INTO RECOMMEND_PATCH (SLIDE_NAME, PATCH_NAME, PATCH_PREDICTION, PATCH_SCORE, X_LOC, Y_LOC, PATCH_GEN_PATH, PATCH_GROUNDTRUTH, ORACLE_SELECTION, ANATOMY, X_SCALE, Y_SCALE) " \
        #     "VALUES ('{slide_name}', '{patch_name}', '{patch_prediction}', '{patch_score}', '{x_loc}', '{y_loc}', '{patch_gen_path}', '{patch_groundtruth}', '{oracle_selection}', '{anatomy}', '{x_scale}', '{y_scale}')"
        # with target.cursor() as cursor:
        #     for patch in all_patches:
        #         patch_name = patch[0] + "_" + patch[3] + "_" + patch[4]
        #         oracle_selection = 1 if patch in all_rec_patches else 0
        #         query_ = query.format(
        #             slide_name = patch[0],
        #             patch_name = patch_name,
        #             patch_prediction = patch[1],
        #             patch_score = patch[2],
        #             x_loc = patch[3],
        #             y_loc = patch[4],
        #             patch_gen_path = "",
        #             patch_groundtruth = "",
        #             oracle_selection = oracle_selection,
        #             anatomy = "Test",
        #             x_scale = 22509,
        #             y_scale = 53169
        #         )
        #         # print(query_)
        #         query_execution(cursor, query_)
        # target.close()
    
    return 

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
