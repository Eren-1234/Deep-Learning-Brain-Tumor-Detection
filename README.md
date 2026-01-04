# 🧠 Brain Tumor Classification using CNN (Beyin Tümörü Sınıflandırma Projesi)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-red)
![Accuracy](https://img.shields.io/badge/Accuracy-80%25-green)

Bu proje, MR (Manyetik Rezonans) görüntülerini kullanarak beyin tümörlerini tespit etmek ve türlerine göre sınıflandırmak amacıyla geliştirilmiş bir Derin Öğrenme (Deep Learning) modelidir. Projede **Evrişimli Sinir Ağları (CNN)** mimarisi kullanılmıştır.

---

## 📋 İçindekiler
- [Proje Özeti](#-proje-özeti)
- [Veri Seti](#-veri-seti)
- [Model Mimarisi](#-model-mimarisi)
- [Sonuçlar ve Performans](#-sonuçlar-ve-performans)
- [Grafik Analizi](#-grafik-analizi)
- [Kurulum ve Kullanım](#-kurulum-ve-kullanım)

---

## 📌 Proje Özeti
Beyin tümörlerinin erken teşhisi, tedavi sürecinde hayati önem taşır. Bu proje, manuel inceleme sürecini hızlandırmak ve radyologlara yardımcı bir karar destek mekanizması sunmak amacıyla geliştirilmiştir. 

Model, görüntüleri 4 farklı sınıfa ayırmaktadır:
1.  **Glioma** (Glioma Tümörü)
2.  **Meningioma** (Meningioma Tümörü)
3.  **Pituitary** (Hipofiz Tümörü)
4.  **No Tumor** (Tümör Yok / Sağlıklı)

---

## 📂 Veri Seti
Projede Kaggle platformunda bulunan **Brain Tumor MRI Dataset** kullanılmıştır.

* **Görüntü Boyutu:** 150x150 piksel (Yeniden boyutlandırılmış)
* **Ön İşleme:** Piksel normalizasyonu (1./255) ve Veri Çoğaltma (Data Augmentation: Rotation, Zoom, Shift) teknikleri uygulanmıştır.
* **Sınıf Dağılımı:** Veri seti Glioma, Meningioma, Notumor ve Pituitary olmak üzere 4 dengeli sınıftan oluşmaktadır.

---

## 🏗 Model Mimarisi
Model, `TensorFlow/Keras` kütüphanesi ile **Sequential** yapıda kurulmuştur:

* **3x Evrişim Bloğu:** Her blokta Conv2D (ReLU aktivasyonu) ve MaxPooling2D katmanları bulunur (32, 64, 128 filtre).
* **Flatten:** Özellik haritalarını tek boyutlu vektöre çevirir.
* **Dense (512):** Tam bağlantılı katman (ReLU).
* **Dropout (0.5):** Aşırı öğrenmeyi (Overfitting) engellemek için nöronların %50'si rastgele kapatılır.
* **Output Layer (4):** 4 sınıf için Softmax aktivasyon fonksiyonu.

---

## 📊 Sonuçlar ve Performans

Model, test verisi üzerinde **%80 Doğruluk (Accuracy)** oranına ulaşmıştır. Detaylı sınıflandırma raporu aşağıdadır:

| Sınıf (Class) | Precision (Kesinlik) | Recall (Duyarlılık) | F1-Score |
| :--- | :---: | :---: | :---: |
| **Glioma** | 0.90 | 0.77 | 0.83 |
| **Meningioma** | 0.76 | 0.53 | 0.63 |
| **No Tumor** | 0.72 | 0.96 | 0.82 |
| **Pituitary** | 0.86 | 0.90 | 0.88 |
| **GENEL BAŞARI** | **0.80** | **0.80** | **0.80** |

### 📉 Hata Metrikleri
Sınıflandırma problemlerinde modelin kararlılığını ölçmek için hesaplanan hata değerleri:

| Metrik | Değer | Açıklama |
| :--- | :--- | :--- |
| **MAE** | 0.1258 | Ortalama Mutlak Hata (Düşük olması iyidir) |
| **MSE** | 0.0762 | Ortalama Kare Hata (Düşük olması iyidir) |
| **RMSE** | 0.2761 | Kök Ortalama Kare Hata |

> **Not:** *MAPE ve SMAPE değerleri, veri setindeki One-Hot Encoding (0 değerlerinin çokluğu) nedeniyle matematiksel sapmaya uğramış ve analiz dışı bırakılmıştır.*

---

## 📈 Grafik Analizi

Eğitim sürecinde elde edilen Accuracy (Başarı) ve Loss (Kayıp) grafikleri aşağıdadır:

<img width="1200" height="600" alt="deneme1" src="https://github.com/user-attachments/assets/389319ef-8c39-4668-8fbb-c8fdc90e052a" />



* **Yorum:** Model, eğitim verisi üzerinde %90 üzeri başarı yakalamıştır. Doğrulama (Validation) verisinde ise %80 bandında dengeli bir seyir izlemiştir. Loss grafiğinde görülen dalgalanmalar, modelin genelleme yaparken zorlandığı noktaları işaret etmektedir.

### Karmaşıklık Matrisi (Confusion Matrix) Yorumu
Modelin en çok karıştırdığı sınıf **Meningioma** olmuştur. Matris incelendiğinde, Meningioma tümörlerinin bir kısmının yanlışlıkla "No Tumor" (Sağlıklı) olarak sınıflandırıldığı görülmüştür. Buna karşın **Pituitary (Hipofiz)** tümörlerinde başarı oranı oldukça yüksektir.

---

## ⚙️ Kurulum ve Kullanım

Projeyi kendi bilgisayarınızda çalıştırmak için:

1.  Repoyu klonlayın:
    ```bash
    git clone [https://github.com/Eren-1234/Deep-Learning-Brain-Tumor-Detection.git](https://github.com/Eren-1234/Deep-Learning-Brain-Tumor-Detection.git)
    ```
2.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install tensorflow matplotlib scipy scikit-learn seaborn
    ```
3.  Veri setini indirip proje klasörüne ekleyin ve kodu çalıştırın:
    ```bash
    python derin_ogrenme.py
    ```

---
*Bu proje Eğitim Amaçlı geliştirilmiştir.*
