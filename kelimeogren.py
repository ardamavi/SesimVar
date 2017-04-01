# Arda Mavi
import data_islem as db
from sklearn import tree

def get_datalar():
    # Eğer database'de tablo yoksa yaratılır:
    db.tablo_yarat('harf_kelime', 'harfler, kelime')
    veriler = db.veri_al("SELECT * FROM 'harf_kelime'")
    return veriler

def harf_list(str_harfler):
    # String'i ondalık sayılardan oluşan oluşan listeye çeviriyor.
    harfler = []
    for harf in str_harfler:
        # Bir harfler dizininden(String) önce harf ondalık sisteme çevriliyor
        # ve yeni sayılar bir liste oluşturuyor:
        harfler.append(ord(harf))
    return harfler

def get_harfler(veriler):
    # Gelen harfler ve kelimeler grubundan harfleri ayırıyor.
    # Daha sonra stringlerden oluşan dizileri harfler listesinden oluşan listeler yapıyor.
    harfler_dizinleri = []
    for veri in veriler:
        # Harfler listesi diğer harfler listesiyle birleştiriliyor:
        harfler_dizinleri.append(harf_list(veri[0]))
    return harfler_dizinleri

def get_kelimeler(veriler):
    # Gelen garfler ve kelimeler grubundan kelimeleri ayırıyor.
    kelimeler = []
    for veri in veriler:
        kelimeler.append(veri[1])
    return kelimeler

def harfler_kelimeler():
    # Database'deki verileri işleyerek harf dizinlerini ve kelimeleri döndürüyor.
    veriler = get_datalar()
    if veriler == []:
        print('Databaseden veri alınamadı!')
        return [], []
    harfler = get_harfler(veriler)
    kelimeler = get_kelimeler(veriler)
    return harfler, kelimeler

def kelime_egitim():
    # Database'deki harf dizileri ve kelime verileri ile yapay zekayı eğitiyor.
    # Database'deki verileri işleyerek harfler ver
    harfler, kelimeler = harfler_kelimeler()
    if harfler == [] or kelimeler == []:
        print('Örnek veriler alınamadı!')
        return
    # Yapay zeka oluşturuluyor:
    clf = tree.DecisionTreeClassifier()
    # Eğitim harfleri ile ve çıktıları olan kelimeler ile yapay zeka eğitiliyor.
    clf = clf.fit(harfler, kelimeler)
    return clf

def kelime_bul(clf,harf_list):
    # Eğitilmiş yapay zeka'ya veriler giriliyor ve sonuç alınıyor:
    if clf == None:
        print('Sınıflandırıcı oluşturulamamış!')
        return
    kelime = clf.predict([harf_list])
    return kelime

print(kelime_bul(kelime_egitim(), harf_list('aada')))

"""
Örnek kullanım:

Doğru_Kelime = kelime_bul(kelime_egitim(), harf_list(Hatalı_Kelime))

kelime_egitim() => Eğitilmiş yapay zekayı dönüyor.
harf_list => String'i harflerin ondalık karşılıklarından oluşan listeyi dönüyor.

"""
