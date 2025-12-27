import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # Görselleştirme için (yüklü değilse: pip install seaborn)

# --- AYARLAR ---
dataset_path = r'C:\Users\hp\Desktop\BeyinTumoruVerisi\Training' # Klasör yolu
img_width, img_height = 150, 150
batch_size = 32
epochs = 20

# 1. VERİ HAZIRLIĞI
print("Veri seti hazırlanıyor...")

# Eğitim için veri çoğaltma (Augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2 
)

# Eğitim Jeneratörü
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Doğrulama Jeneratörü (Eğitim sırasında başarım ölçmek için)
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 2. MODEL MİMARİSİ
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax') # 4 Sınıf
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3. MODELİN EĞİTİLMESİ
print("Model eğitimi başlıyor...")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# 4. GRAFİKLER (ACCURACY & LOSS)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Eğitim Başarımı')
plt.plot(epochs_range, val_acc, label='Doğrulama Başarımı')
plt.legend(loc='lower right')
plt.title('Eğitim ve Doğrulama Başarımı')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Eğitim Kaybı')
plt.plot(epochs_range, val_loss, label='Doğrulama Kaybı')
plt.legend(loc='upper right')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.show()

# -----------------------------------------------------------
# 5. DETAYLI MODEL DEĞERLENDİRMESİ (TÜM METRİKLER)
# -----------------------------------------------------------
print("\n--- DETAYLI PERFORMANS RAPORU ---")

# Test verisini sıralı (shuffle=False) olarak tekrar yüklüyoruz
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_generator = test_datagen.flow_from_directory(
    dataset_path, # Yukarıda tanımladığın klasör yolu
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False # ÖNEMLİ: Gerçek ve Tahmin sırası karışmasın diye
)

# Tahminleri Al
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1) # Modelin seçtiği sınıf (0,1,2,3)
y_true = test_generator.classes         # Gerçek sınıf (0,1,2,3)
class_labels = list(test_generator.class_indices.keys())

# 1. Classification Report (F1, Precision, Recall)
print("\n>>> SINIFLANDIRMA TABLOSU (Precision, Recall, F1):")
print(classification_report(y_true, y_pred, target_names=class_labels))

# 2. Confusion Matrix
print("\n>>> KARMAŞIKLIK MATRİSİ:")
print(confusion_matrix(y_true, y_pred))

# 3. Hata Oranları (MAE, MSE, RMSE, MAPE, SMAPE)
# Hata hesabı için 'One-Hot' formatına çeviriyoruz (Örn: Sınıf 2 -> [0, 0, 1, 0])
y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=4)

# Küçük bir sayı (epsilon) ekliyoruz ki 0'a bölünme hatası olmasın
epsilon = 1e-10 

# MAE & MSE & RMSE
mae = mean_absolute_error(y_true_one_hot, predictions)
mse = mean_squared_error(y_true_one_hot, predictions)
rmse = np.sqrt(mse)

# MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_true_one_hot - predictions) / (y_true_one_hot + epsilon))) * 100

# SMAPE (Symmetric Mean Absolute Percentage Error)
# Formül: 2 * |y_pred - y_true| / (|y_true| + |y_pred|)
pay = np.abs(predictions - y_true_one_hot)
payda = (np.abs(y_true_one_hot) + np.abs(predictions)) + epsilon
smape = 100 * np.mean(2.0 * pay / payda)

print("\n>>> HATA METRİKLERİ (Sayısal Farklılıklar):")
print(f"Mean Absolute Error (MAE)            : {mae:.5f}")
print(f"Mean Squared Error (MSE)             : {mse:.5f}")
print(f"Root Mean Squared Error (RMSE)       : {rmse:.5f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f} %")
print(f"Symmetric MAPE (SMAPE)               : {smape:.2f} %")

print("\n------------------------------------------------")
print("Tüm hesaplamalar tamamlandı.")