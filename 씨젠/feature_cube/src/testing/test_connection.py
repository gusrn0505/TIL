import project_config as cfg

import pymysql

def conn():
    mydb=pymysql.Connect(host=cfg.mariadb["host"],user=cfg.mariadb["user"],password=cfg.mariadb["password"],db=cfg.mariadb["db"],autocommit=True)
    return mydb.cursor()

def db_exe(query,c):
    try:
        if c.connection:
            print("connection exists")
            c.execute(query)
            return c.fetchall()
        else:
            print("trying to reconnect")
            c=conn()
    except Exception as e:
        return str(e)

dbc=conn()
print(db_exe("select * from ai_predictions_copy LIMIT 1",dbc))


