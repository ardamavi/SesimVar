# Arda Mavi
import cv2
import numpy as np
import os
import sys
import platform
from multiprocessing import Process
import data_islem as di
import harfogren as ho

# Debugging için ayarlar:
harfogren_var = bool(int(sys.argv[1])) # Datalar girilene kadar kapalı kalmalı.
# Programa gelen argüman string olarak alınır tam sayıya çevrilir ve bool yapılır.
# Programa gelen ilk argümanlar:
# Eğitim süreci ise = 0
# Yapay zeka ile çalışma anında = 1

tum_yuzleri_algilama = False
seslendirme_yap = True
gozluklu = True

# İşletim sistemine göre ekranı temizleyen fonksiyonumuz:
def clear_screen():
    # clear_screen Screen:
    if(platform.system() == 'Linux' or 'Darwin'):
        os.system('clear')
    else:
        os.system('cls')
    return

def seslendir(text):
    if(platform.system() == 'Darwin'):
        arg = 'say {0}'.format(text)
        Process(target=os.system, args=(arg,)).start()
    else:
        print('Seslendirme şimdilik sadece Darwin kabuğunda çalışmaktadır.')
    return

# harfogren aktif ve database boş değil ise True:
def hoT_dbT(kullanici_adi):
    if kullanici_adi != '':
        kullanici_adi = 'WHERE kullanici="{0}"'.format(kullanici_adi)
    return di.veri_al('SELECT * FROM agiz_harf {0}'.format(kullanici_adi)) != [] and harfogren_var

def data_kayit(kullanici_adi, agiz_data, clf):
    clear_screen()
    print('Anlayamadım :(\nBu söylediğiniz hangi harf nedir?\nNot: Eğer bir harf belirtmek istemiyorsanız \'x\' giriniz.')
    # Girilecek harfi tutacağımız değişkeni oluşturuyoruz:
    harf = ''
    while 1:
        # Kullanıcıdan bir harf alınır:
        harf = input('Söylediğiniz harf: ')
        # Veri girildi mi kontrol edilir:
        if harf != '':
            # Veriler database'de küçük harf şekline tutuluyor.
            # Bunun için girilen harf her zaman küçük harf olmalı.
            # Bu yüzden girilen harfin küçük harf karşılığı tutuluyor.
            harf = harf[0].lower()
            break
        else:
            print('Hatalı Girş!')
    # Veri Database'mize kaydoluyor:
    di.data_ekle('agiz_harf', 'kullanici, harf, agiz_genislik, agiz_yukseklik', (kullanici_adi, harf, agiz_data[0], agiz_data[1]))
    if harfogren_var:
        clf = ho.db_egitim(kullanici_adi)
    return clf

def harf_al(kullanici_adi, agiz_data, clf):
    # agiz_data = [Genişlik, Yükseklik]
    if hoT_dbT(kullanici_adi):
        gelen_veri = ho.getHarf(clf,agiz_data[0],agiz_data[1])
        clear_screen()
    else:
        if kullanici_adi != '':
            kullanici_adi = ' kullanici="{0}" AND'.format(kullanici_adi)
        #  “data_islemler” dosyamızdan “veri_al” fonksiyonunu kullanarak kullanıcı adı ve ağız yapısına uygun harfi alıyoruz:
        gelen_veri = di.veri_al("SELECT harf FROM agiz_harf WHERE agiz_genislik={1} AND agiz_yukseklik={2}".format(kullanici_adi, agiz_data[0], agiz_data[1]))
        # Fonksiyonumuz  “gelen_veri” değişkenini döndürür:
    return gelen_veri

def goruntu_isleme(kullanici_adi, clf):
    # OpenCV algoritmamızla çalıştıracağımız eğitilmiş dosyalarımızı belirliyoruz:
    yuz_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

    # Debugging için:
    if gozluklu:
        goz_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    else:
        goz_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')

    agiz_cascade = cv2.CascadeClassifier('data/haarcascades/mouth2.xml')

    # Görüntüyü alacağımız kameramızı tanımlıyoruz:
    cap = cv2.VideoCapture(0)
    while 1:
        # İleride kullanmak için ağız değişkeni oluşturuyoruz:
        agiz = [-1,-1]
        # Tanımladığımız kameradan bir fotoğtaf çekiyoruz ve alıyoruz:
        ret, img = cap.read()

        # Yüz tanıma sistemimizin daha hızlı çalışması için fotoğrafı gri renk uzayına çeviriyoruz:
        # RGB(3 katmanlı pixel matrixi) sisteminden gri renk uzayına(tek katmanlı pixel matrixine) çeviriyoruz:
        # Nesne algılama sistemimizi, gri renk uzayındaki(tek katmanlı pixel matrixindeki) fotoğrafımızda yapacağız.
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Eğitilmiş dosyamızla OpenCV nesne bulma algoritmamızı çalıştırıyoruz:
        # Eğer yüzler bulunursa yüzlerin konumlarını fonksiyonumuz bize veriyor:
        yuzler = yuz_cascade.detectMultiScale(grayImg)
        kac_yuz_algilama = 1
        # Kaç yüz algılanmak isteniyor belirliyoruz:
        if tum_yuzleri_algilama:
            kac_yuz_algilama = len(yuzler)
        # Gelen yüzler içerisinde geziyoruz:
        for (x,y,w,h) in yuzler[0:kac_yuz_algilama]:
            # Yüzün bulunduğu alanı opencv kütüphanemiz ile bir dörtgen içine alıyoruz:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,230,0),2)
            # Yüzün konumunu kaydediyoruz:
	        # İleride göz ve ağızı bulmak için bu verileri kullanacağız:
            yuz = img[y:y+h, x:x+w]
            grayYuz = grayImg[y:y+h, x:x+w]

            # Eğitilmiş dosyamızla OpenCV nesne bulma algoritmamızı çalıştırıyoruz.
            # Ve bulunan göz koordinatlarını kaydediyoruz:
            gozler = goz_cascade.detectMultiScale(grayYuz)
            # Ağızı bulurken kullanmamız gereken bir değişken oluşturuyoruz:
            göz_lerin_alti_y = 0
            # Bulunan gözler içerisinde geziyoruz:
            for (ex,ey,ew,eh) in gozler[0:2]:
                # Gözün bulunduğu alanı opencv kütüphanemiz ile bir dörtgen içine alıyoruz:
                cv2.rectangle(yuz,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)
                # İleride kullanmak için aşağıda olan gözün altının y koordinatını kaydediyoruz:
                if göz_lerin_alti_y < ey+eh:
                    göz_lerin_alti_y = ey+eh

            # Göz bulamadıysa tekrar ara:
            if len(gozler) < 1 and göz_lerin_alti_y > 0:
                continue

            # Ağız sadece gözlerin altında olduğu için gözlerin üstünü aramamız performansı düşürebilir..
            # Bu yüzden ağızı sadece gözlerin altında aramalıyız:
            goz_alti = img[y+göz_lerin_alti_y+1:y+h, x:x+w]
            grayGoz_alti = grayImg[y+göz_lerin_alti_y+1:y+h, x:x+w]

            # Ağız bölgesi çizilir:
            cv2.rectangle(img,(x+1,y+göz_lerin_alti_y+1),(x+w-1,y+h-1),(255,255,255),1)

            # Eğitilmiş dosyamızla OpenCV nesne bulma algoritmamızı çalıştırıyoruz.
            # Ve bulunan ağız koordinatlarını kaydediyoruz:
            agizlar = agiz_cascade.detectMultiScale(grayGoz_alti)
            # Bulunan ağızlar içerisinde geziyoruz:
            for (mx,my,mw,mh) in agizlar[0:1]:
                # Ağızı dörtgen içine alıyoruz:
                cv2.rectangle(goz_alti,(mx,my),(mx+mw,my+mh),(255,255,0),2)
                # İleride kullanmak için ağız genişliğini ve yüksekliğini kaydediyoruz:
                # agiz = [Genişlik, Yükseklik]
                agiz = [int((mx+mw)-mx),int((my+mh)-my)]

        # Seslendirme yapılacak ise ve ağız bulunmuşsa:
        if seslendirme_yap and agiz != [-1,-1]:
            while 1:
                cv2.destroyAllWindows()
                harfler = harf_al(kullanici_adi, agiz, clf)
                if harfler != [] and harfogren_var == True:
                    # Fotoğrafın son halini gösteriyoruz:
                    cv2.imshow('Sesim Var!',img)
                    # Eğer kişinin ağızı kapalıysa 'x' değeri döneceğinden seslendirme yapılmaz:
                    if harfler[0][0] != 'x':
                        seslendir(harfler[0][0])
                    break
                else:
                    cv2.putText(img,'Bilinmeyen Kelime !', (150,200), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255))
                    cv2.imshow('Sesim Var!',img)
                    clf = data_kayit(kullanici_adi, agiz, clf)
                    if harfogren_var == False:
                        break

        else:
            cv2.imshow('Sesim Var!',img)

        if cv2.waitKey(1) == 27: # Decimal 27 = Esc
            break
    # Kamerayı kapatıyoruz:
    cap.release()

    # Fotoğrafı kullanıcıya gösterirken kullandığımız uygulama penceresini kapatıyoruz:
    cv2.destroyAllWindows()
    return

def giris_yap():
    print('Not: Üyeliğiniz yoksa yeni üyelik oluşturulur!')
    print('Not: Kullanıcı odaklı olmasını istemiyorsanzı boş bırakınız!')
    kullanici_adi = input('Kullanıcı Adı: ')
    clear_screen()
    return kullanici_adi

def main():
    clear_screen()
    print('Sesim Var!\nArda Mavi -ardamavi.com\n\nÇıkış: Esc\nÇıkmaya Zorla: kntrl+c\n')
    # Eğer DataBase veya tablo yok ise yeni process ile yaratılır.
    Process(target=di.tablo_yarat, args=('agiz_harf','kullanici, harf, agiz_genislik, agiz_yukseklik')).start()
    kullanici_adi = giris_yap()
    clf = ''
    if hoT_dbT(kullanici_adi):
        clf = ho.db_egitim(kullanici_adi)
    goruntu_isleme(kullanici_adi, clf)
    clear_screen()
    print('Sesim Var! - Program Sonlandırıldı !\nArda Mavi -ardamavi.com\n')
    return

if __name__ == "__main__":
    main()
