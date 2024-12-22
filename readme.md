# README

## Proje Özeti
Bu proje, özel bir veri seti üzerinde görüntü sınıflandırması yapmak için bir Konvolüsyonel Sinir Ağı (CNN) oluşturmayı ve değerlendirmeyi kapsamaktadır. Proje, veri ön işleme, veri artırma ve farklı koşullar altında değerlendirme tekniklerini içermektedir.

---

## Veri Seti
- **Kaynak Dizin**: `C:\JPEGImages`
- **Filtrelenmiş Veri Seti Dizini**: `filtered_dataset`
- **Seçilen Sınıflar**: `collie`, `dolphin`, `elephant`, `fox`, `moose`, `rabbit`, `sheep`, `squirrel`, `giant panda`, `polar bear`
- **Görüntü Ön İşleme**:
  - Görseller `128x128` piksel boyutlarına yeniden boyutlandırıldı.
  - Değerler `[0, 1]` aralığında normalize edildi.
- **Sınıf Dağılımı**:
  - Her sınıfta 650 görüntü bulunuyor.

---

## Kullanılan Araçlar ve Kütüphaneler
- **Python Standart Kütüphaneleri**: `os`, `shutil`
- **Veri İşleme**: `numpy`
- **Görüntü İşleme**: `cv2` (OpenCV)
- **Makine Öğrenimi**: `TensorFlow/Keras`
- **Veri Artırma**: `ImageDataGenerator` (Keras)
- **Görselleştirme**: `matplotlib`

---

## Model Mimarisi
TensorFlow/Keras kullanılarak oluşturulan CNN modeli şu katmanlardan oluşmaktadır:
1. **Giriş Katmanı**: `128x128x3` boyutundaki görüntüleri alır.
2. **Konvolüsyonel Katmanlar**:
   - Artan filtre sayılarıyla (`32`, `64`, `128`) üç Conv2D katmanı ve ReLU aktivasyonu.
   - Her Conv2D katmanından sonra MaxPooling2D ile uzaysal boyutların azaltılması.
3. **Düzleştirme Katmanı**: 2D özellik haritalarını 1D vektöre dönüştürür.
4. **Yoğun Katmanlar**:
   - `128` nöronlu ve ReLU aktivasyonlu tam bağlantılı bir katman.
   - Aşırı öğrenmeyi azaltmak için %50 oranında Dropout katmanı.
   - Çıkış katmanı: `10` sınıf için softmax aktivasyonu.

---

## Veri Artırma
Eğitim setine `ImageDataGenerator` kullanılarak veri artırma işlemleri uygulanmıştır:
- **Döndürme**: Maksimum 20 derece.
- **Kaydırma**: Yatay ve dikey eksende maksimum %20.
- **Yakınlaştırma**: Maksimum %20.
- **Aynalama**: Yatay aynalama.
- **Kesme**: Maksimum %20.
- **Doldurma Modu**: Boş pikseller için en yakın komşu.

---

## Değerlendirme
### Test Setleri
1. **Orijinal Test Seti**
   - Veri setinden doğrudan ayrılmış test görüntüleri.

2. **Manipüle Edilmiş Test Seti**
   - Görüntülere aşağıdaki işlemler uygulanmıştır:
     - Gauss Bulanıklığı
     - Yatay Aynalama
     - Rastgele Gürültü Ekleme

3. **Renk Sabitliği (WB) Uygulanan Test Seti**
   - Gray World Algoritması ile renk sabitliği sağlanmıştır.
   - Genelleştirmeyi artırmak için normalize edilmiştir.

### Ölçütler
- **Kayıp Fonksiyonu**: `Sparse Categorical Crossentropy`
- **Performans Ölçütü**: Doğruluk

### Sonuçlar
Model performansı farklı test setlerinde değerlendirilmiştir:
- **Orijinal Test Seti Doğruluğu**: `XX%`
- **Manipüle Edilmiş Test Seti Doğruluğu**: `XX%`
- **Renk Sabitliği Test Seti Doğruluğu**: `XX%`

---

## Sonuçların Görselleştirilmesi
Performans sonuçları bir çubuk grafik ile görselleştirilmiştir. Bu grafik, manipülasyon ve ön işlemenin modelin dayanıklılığı üzerindeki etkisini açıkça göstermektedir.

---

## Çalıştırma Adımları
1. **Ortamı Hazırlayın**:
   - Gerekli kütüphaneleri yüklemek için `pip install -r requirements.txt` komutunu çalıştırın.
   - Veri setini belirtilen klasör yapısına yerleştirin.

2. **Skripti Çalıştırın**:
   - Ana scripti çalıştırarak verileri ön işleyin, modeli eğitin ve sonuçları değerlendirin.

3. **Çıktılar**:
   - Eğitilmiş model ve değerlendirme metrikleri kaydedilecektir.
   - Manipüle edilmiş ve WB test setleri ilgili dizinlere kaydedilecektir.

---

## Gelecek Çalışmalar
- Ek veri artırma tekniklerinin keşfi.
- Transfer öğrenimi kullanılarak doğruluğun artırılması.
- Doğruluk, geri çağırma ve F1-skoru gibi gelişmiş değerlendirme metriklerinin uygulanması.

