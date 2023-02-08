import pymysql
import pandas as pd
from datetime import datetime, timedelta

source_db = {
    "host": "219.252.39.14",
    "user": "root",
    "password": "seegeneai2020", 
    "db": "seegene_il",
    "table_1": "slides_queue",
    "table_2": "tb_test_result",
}


target_db = {
    "host": "219.252.39.14",
    "user": "root",
    "password": "seegeneai2020",
    "db": "seegene_il",
    "table": "recommend_slide",    
}

# N은 모델을 통한 예측과 전문가의 판단이 서로 다른 경우에만 진행. 
# D와 M 이면 모두 포함. 
QUERY = "SELECT SQ.ANATOMY, SQ.SLIDE_NAME, SQ.SLIDE_PATH, SQ.DATE_TIME_ADDED, SQ.SLIDE_LEVEL_LABEL AS GT, TR.RESULT_TYPE AS AI_PRED, SQ.LABEL_P "\
    "FROM TB_TEST_RESULT AS TR LEFT JOIN SLIDES_QUEUE AS SQ ON TR.BARCODE = SQ.SLIDE_NAME AND TR.SLIDE_TYPE = SQ.ANATOMY "\
    "WHERE SQ.ANATOMY = '{anatomy}' AND SQ.SLIDE_LEVEL_LABEL = '{condition}' AND {mismatch} AND {time_range} "\
    "ORDER BY SQ.LABEL_P ASC LIMIT 5"


def main():
    connection = connect_to_db(source_db)
    # cursor : 하나의 DB Connection에 대해서 독립적으로 SQL 문을 실행할 수 있는 작업환경을 제공하는 객체. 
    # Cursor를 통해서 SQL 문을 실행할 수 있으며, 응용 프로그램이 실행결과를 튜플 단위로 접근할 수 있도록 함 

    # with : 자원을 획득하고 사용하고, 반납할 때 주로 사용한다. DB에 연결했다가 할 일을 마치고 닫는 듯. 
    with connection.cursor() as cursor:
        recommended_slides = []
        for anatomy in ['Colon', 'Stomach']:
            for condition in ['N', 'D', 'M']: # 3 Class 
                # start_date = yesterday; end_date = today

                # timedelta(1) : 지금 시점에서 하루 더하기  
                start_date = str(datetime.strftime((datetime.now() - timedelta(1)), '%Y-%m-%d')) # e.g."2022-05-01" 
                end_date = str(datetime.strftime(datetime.now(), '%Y-%m-%d'))

                start_date = "2022-08-01"
                end_date = "2022-08-02"

                time_range = "DATE(SQ.DATE_TIME_ADDED) BETWEEN '" + start_date + "' AND '" + end_date + "'"

                # mismatch의 역할은 뭐지? N이 의미하는 바는 Normal이 맞겠지? mismatch의 조건을 부여한 건가? 그럼 '1=1'의 의미는? 
                if condition == 'N':
                    mismatch = "SQ.SLIDE_LEVEL_LABEL != TR.RESULT_TYPE"
                    # slide_level_label 과 tr.result_type 은 어떻게 다른 것인가? 
                    # 하나는 모델의, 하나는 전문가의 판단으로 보이는데. 즉, 서로의 판단이 다른 것에 대해서 불러오기. 
                    # 노말인데 두 판단이 다를 경우에만 mismatch 하다. 

                else:
                    # 이건 True 임을 보장하기 위해서 이렇게 넣은 건가? 걍 다 불러오기. 
                    mismatch = '1 = 1'
                
                # format. 위에 작성한 Query 문에서 빈칸을 채워줌. condition / Anatomy/ mismatch 조건에 맞는 데이터를 불러와라.  
                query = QUERY.format(condition=condition, anatomy=anatomy, mismatch=mismatch, time_range = time_range) 
                result = query_execution(cursor, query) # 이때 cursor은 connection.cursor()을 의미. source_db의 정보를 불러온 것에 대해서 SQL문을 실행할 수 있게 된 객체. 연결된 DB에 대해서 query 명령문을 보냄. 
                recommended_slides.extend(result) # DB에 query 명령문을 보낸 결과물을 저장 
    connection.close()

    # insert
    # target = connect_to_db(target_db)
    # query = "INSERT INTO RECOMMEND_SLIDE (MODEL_KEY, SLIDE_NAME, ANATOMY, SLIDE_PATH, SLIDE_COPY_PATH, SCAN_DATE, WSI_PREDICTION, WSI_SCORE, SLIDE_GROUNDTRUTH, ORACLE_SELECTION) "\
    #     "VALUES ('{model_key}', '{slide_name}', '{anatomy}', '{slide_path}', '{slide_copy_path}', '{scan_date}', '{prediction}', '{score}', '{ground_truth}', '{oracle_selection}')"
    # with target.cursor() as cursor:
    #     for recommended_slide in recommended_slides:
    #         slide_path = recommended_slide[2].replace('\\', '/')
    #         query_ = query.format(
    #             model_key='learn_test',
    #             anatomy=recommended_slide[0],
    #             slide_name=recommended_slide[1],
    #             slide_path=slide_path,
    #             slide_copy_path=slide_path,
    #             scan_date=recommended_slide[3],
    #             ground_truth=recommended_slide[4],
    #             prediction=recommended_slide[5],
    #             score=recommended_slide[6],
    #             oracle_selection=0
    #         )
    #         query_execution(cursor, query_)
    # target.close()
            

    # df = pd.DataFrame(recommended_slides, columns=['anatomy', 'slide_name', 'slide_path', 'date_time_added', 'label', 'prediction', 'label_p'])
    # print(pd.pivot_table(df, values='slide_name', index='anatomy', columns='label', aggfunc='count'))
    
    return recommended_slides


def connect_to_db(db_config: dict):
    connection = pymysql.Connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        db=db_config['db'],
        autocommit=True)
    return connection


def query_execution(cursor, query: str):
    cursor.execute(query) # query 명령어를 실행한다. 
    return cursor.fetchall() # 커서의 fetchall() 메서드는 모든 데이타를 한꺼번에 클라이언트로 가져올 때 사용된다. 


if __name__ == '__main__':
    main()