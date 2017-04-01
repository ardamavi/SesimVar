# Arda Mavi
import os
import numpy as np
import data_islem as db
from sklearn.neural_network import MLPClassifier

def veriler(kullanici):
    # Sinir ağlarının eğitim sürecinde kullanılacak dataların çekilmesi:
    if kullanici != '':
        kullanici = 'WHERE kullanici="{0}"'.format(kullanici)
    db_veriler = db.veri_al('SELECT agiz_genislik, agiz_yukseklik FROM agiz_harf {0}'.format(kullanici))
    db_etiketler = db.veri_al('SELECT harf FROM agiz_harf {0}'.format(kullanici))

    # Çıktı değerlerinin(etiketlerin) sinir ağlarının anlayacağı şekilde yapılandırması:
    # Databaseden gelen [(x,),(x,)] şeklinde veriler, ['x','x'] şekline çevrilir
    db_etiketler = np.ravel(db_etiketler)

    return db_veriler, db_etiketler

def db_egitim(kullanici):
    X, y = veriler(kullanici)

    # Sinir Ağlarının oluşturulması:
    # MLPClassifier -> multi-layer perceptron (MLP)
    # hidden_layer_sizes verilere göre değişiklik gösterebilir.
    clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5,3))

    # TODO: Sİnir ağları yapılandırılacak

    # Sinir ağlarının eğitimi:
    clf = clf.fit(X, y)
    return clf

def getHarf(clf,w,h):
    # Sinir ağlarını kullanarak ağız yapısından harf tahmini:
    harf = clf.predict([[w,h]])
    print('Sinir Ağları Harf Tahmini: ', harf)
    return harf
