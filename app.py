import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 1. Modeli Masaüstünden Yükle
# Masaüstü yolunu otomatik bulur
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
model_yolu = os.path.join(desktop_path, 'beyin_tumoru_modeli.h5')

print(f"Model yükleniyor: {model_yolu}")
model = tf.keras.models.load_model(model_yolu)

# Sınıf isimleri (Klasörlerinle aynı sırada)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def tahmin_et(img):
    if img is None:
        return None

    # 2. Ön İşleme (Preprocessing)
    # Eğitimdekiyle birebir aynı işlemleri yapıyoruz
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 3. Tahmin
    predictions = model.predict(img_array)
    
    # Sonuçları sözlük yapısına çevir (Gradio için)
    sonuc_sozlugu = {class_names[i]: float(predictions[0][i]) for i in range(4)}
    
    return sonuc_sozlugu

# 4. Arayüz Tasarımı
arayuz = gr.Interface(
    fn=tahmin_et, 
    inputs=gr.Image(type="pil", label="MR Görüntüsünü Buraya Sürükle"), 
    outputs=gr.Label(num_top_classes=4, label="Yapay Zeka Tahmini"),
    title="🧠 Beyin Tümörü Tespit Sistemi",
    description="Model şu an bilgisayarında çalışıyor! Bir beyin MR görüntüsü yükle ve sonucu gör.",
    theme="soft"
)

# 5. Başlat (Share=True ile herkese açık link verir)
arayuz.launch(share=True)