# Arda Mavi
import os
import data_islem as db
from sklearn import tree

def db_egitim(kullanici):
    if kullanici != '':
        kullanici = 'WHERE kullanici="{0}"'.format(kullanici)
    db_veriler = db.veri_al('SELECT agiz_genislik, agiz_yukseklik FROM agiz_harf {0}'.format(kullanici))
    db_etiketler = db.veri_al('SELECT harf FROM agiz_harf {0}'.format(kullanici))

    # Alınan verime bağlı olarak sınıflandırma yöntemi değiştirilebilir.

    # Farklı sınıflandırma yöntemleri ve verilere göre alınan performansları grafikler şeklinde
    # projemiz içerisinde 'Ekler' klasöründe 'classification.png' içerisinde gösterilmiştir.

    # Sınıflandırma yöntemleri arasından, elimdeki verileri(eğitim verilerini) kullanarak denediğim,
    # sınıflandırma işlemlerinden en iyi performansı aldığım yöntemi seçtim.
    # Fakat yöntem seçimi sonuçlardan alınan verime göre değiştirilebilir.

    # Projemizde Scikit(sklearn) kütüphanemizle kullandığımız sınıflandırıcı: DecisionTreeClassifier
    # DecisionTreeClassifier = Karar Ağacı Sınıflandırıcısı

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(db_veriler, db_etiketler)

    return clf

def getHarf(clf,w,h):
    harf = clf.predict([[w,h]])
    print('Karar Ağacı Harf Tahmini: ', harf)
    return harf
