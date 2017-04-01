# Arda Mavi
import sqlite3

# Database bağlantısı kurmak:
def set_sql_connect(database_name):
    return sqlite3.connect(database_name)

# Database Cursor ayarlamak:
def set_sql_cursor(database_connect):
    return database_connect.cursor()

# Database bağlantısı kurup Cursor ayarlamak:
def set_connect_and_cursor():
    vt = set_sql_connect('data/database/database.sqlite')
    db = set_sql_cursor(vt)

    return vt, db

# Database bağlanrısını kapatmak için:
def close_connect(vt):
    if vt:
        vt.commit()
        vt.close

# Database'de yaratılmamışsa yeni tablo yaratmak için:
def tablo_yarat(table_name, columns):
    vt, db = set_connect_and_cursor()
    db.execute("CREATE TABLE IF NOT EXISTS {0} ({1})".format(table_name, columns))
    close_connect(vt)

# Database'den veri almak için:
def veri_al(sql_komut):
    vt, db = set_connect_and_cursor()
    db.execute(sql_komut)
    gelen_veri = db.fetchall()
    close_connect(vt)
    return gelen_veri

# Database'e veri eklemek için:
def data_ekle(table, eklenecek_sutun, eklenecek):
    vt, db = set_connect_and_cursor()
    db.execute("INSERT INTO '{0}'({1}) VALUES {2}".format(table, eklenecek_sutun, eklenecek))
    # eklenecek_sutun örnek: 'sütun1','sütun2'
    # eklenecek örnek: data1, data2
    close_connect(vt)

# Database'deki verileri güncellemek için:
def data_guncelle(table, nerden, nasil):
    vt, db = set_connect_and_cursor()
    db.execute("UPDATE {0} SET {2} WHERE {1}".format(table, nerden, nasil))
    close_connect(vt)
