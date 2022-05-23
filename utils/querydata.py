import utils.dbconnection as mypdbc

def queryVideoPath(DataBase, fieldid):
    db = mypdbc.DBConn01(DataBase=DataBase)
    sql = 'select * from ' + fieldid + '_record order by id desc limit 1'
    videopath = ''
    try:
        cursor = db.cursor()
        cursor.execute(sql)
        data_list = cursor.fetchall()
        videopath = data_list[0]['videopath']
    except Exception as e:
        print("Exception occured:{}".format(e))
    finally:
        db.close()
    return videopath

