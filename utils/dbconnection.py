import pymysql

def DBConn01(DataBase, username='sclab520', password='sclab520', IP='140.121.136.63', charSet = 'utf8mb4'):
    cusrorType = pymysql.cursors.DictCursor
    db = pymysql.connect(host=IP, user=username, password=password, db=DataBase, charset=charSet,
                         cursorclass=cusrorType)
    return db
