import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

selected_classes = ["collie", "dolphin", "elephant", "fox", "moose",
                    "rabbit", "sheep", "squirrel", "giant panda", "polar bear"]


input_path = "C:\\JPEGImages"  # JPEGImages klasörünüzün yolu
output_path = "filtered_dataset"  # Seçilen sınıflar için yeni klasör


# Çıkış klasörünü oluştur
os.makedirs(output_path, exist_ok=True)

# Her sınıf için işlemleri gerçekleştir
for cls in selected_classes:
    class_path = os.path.join(input_path, cls)
    target_path = os.path.join(output_path, cls)
    os.makedirs(target_path, exist_ok=True)
    
    # İlk 650 resmi seç ve yeni klasöre taşı
    images = os.listdir(class_path)[:650]
    for img in images:
        shutil.copy(os.path.join(class_path, img), target_path)

print("Sınıflar hazırlandı ve dosyalar kopyalandı.")


# Resim boyutu
image_size = (128, 128)

def preprocess_images(image_path, target_size):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, target_size)  # Resim boyutlandırma
    img_normalized = img_resized / 255.0  # Normalize
    return img_normalized


X = []  # Görüntüler
y = []  # Etiketler

for idx, cls in enumerate(selected_classes):
    class_path = os.path.join(output_path, cls)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        X.append(preprocess_images(img_path, image_size))
        y.append(idx)  # Etiketler: sınıfın indeksi

X = np.array(X)
y = np.array(y)

print(f"Toplam Görüntü Sayısı: {len(X)}")


# %70 Eğitim, %30 Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Eğitim Seti Boyutu: {len(X_train)}")
print(f"Test Seti Boyutu: {len(X_test)}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Modeli oluştur
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # Giriş katmanı
    MaxPooling2D(pool_size=(2, 2)),  # Havuzlama
    Conv2D(64, (3, 3), activation='relu'),  # İkinci katman
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),  # Üçüncü katman
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),  # Katmanları düzleştirme
    Dense(128, activation='relu'),  # Tam bağlantılı katman
    Dropout(0.5),  # Aşırı öğrenmeyi azaltmak için
    Dense(len(selected_classes), activation='softmax')  # Çıkış katmanı (10 sınıf için)
])

# Modeli derle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Model oluşturuldu.")


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Veri artırma işlemleri
datagen = ImageDataGenerator(
    rotation_range=20,       # Rastgele döndürme
    width_shift_range=0.2,   # Genişlik kaydırma
    height_shift_range=0.2,  # Yükseklik kaydırma
    shear_range=0.2,         # Kesme dönüşümü
    zoom_range=0.2,          # Yakınlaştırma
    horizontal_flip=True,    # Yatay çevirme
    fill_mode='nearest'      # Boş alanları doldurma
)

datagen.fit(X_train)  # Eğitim setine veri artırmayı uygula

# Modeli eğit
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),  # Veri artırma ile eğitim
    validation_data=(X_test, y_test),              # Test doğrulama
    epochs=10                                      # Eğitim döngüsü sayısı
)

print("Model eğitimi tamamlandı.")


# Modeli test et
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Seti Doğruluğu: {test_accuracy * 100:.2f}%")






# 2. Manipüle Edilmiş Test Seti


def get_manipulated_images(image_set, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    manipulated_images = []
    for i, img in enumerate(image_set):
        manipulated_img = cv2.GaussianBlur(img, (5, 5), 0)  # Örnek: Bulanıklaştırma
        manipulated_img = cv2.flip(manipulated_img, 1)  # Örnek: Yatay çevirme
        manipulated_img = cv2.add(manipulated_img, np.random.normal(0, 15, manipulated_img.shape))  # Gürültü ekleme
        manipulated_images.append(manipulated_img)
        # Manipüle edilmiş görselleri kaydetmek için
        cv2.imwrite(os.path.join(output_dir, f"img_{i}.jpg"), manipulated_img * 255)
    return np.array(manipulated_images)

# Manipüle edilmiş test setini oluştur
manipulated_test_images = get_manipulated_images(X_test, "manipulated_test_set")

# Manipüle edilmiş test setini model üzerinde test et
manipulated_test_loss, manipulated_test_accuracy = model.evaluate(manipulated_test_images, y_test)
print(f"Manipüle Edilmiş Test Seti Doğruluğu: {manipulated_test_accuracy * 100:.2f}%")

# 3. Renk Sabitliği Uygulanmış Test Seti
def get_wb_images(image_set, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    wb_images = []
    for i, img in enumerate(image_set):
        mean_channels = np.mean(img, axis=(0, 1))  # Renk kanallarının ortalaması
        wb_img = img / mean_channels  # Gray World algoritması
        wb_img = wb_img / np.max(wb_img)  # Normalize et
        wb_images.append(wb_img)
        # Renk sabitliği uygulanmış görselleri kaydetmek için
        cv2.imwrite(os.path.join(output_dir, f"wb_img_{i}.jpg"), wb_img * 255)
    return np.array(wb_images)

# Renk sabitliği uygulanmış test setini oluştur
wb_test_images = get_wb_images(manipulated_test_images, "wb_test_set")

# Renk sabitliği uygulanmış test setini model üzerinde test et
wb_test_loss, wb_test_accuracy = model.evaluate(wb_test_images, y_test)
print(f"Renk Sabitliği Uygulanmış Test Seti Doğruluğu: {wb_test_accuracy * 100:.2f}%")

# Sonuçları karşılaştırma
results = {
    "Orijinal Test Seti": test_accuracy * 100,
    "Manipüle Edilmiş Test Seti": manipulated_test_accuracy * 100,
    "Renk Sabitliği Uygulanmış Test Seti": wb_test_accuracy * 100
}

# Tüm sonuçları yazdır
for test_set, accuracy in results.items():
    print(f"{test_set}: {accuracy:.2f}%")

# Performansı grafikleştirme
import matplotlib.pyplot as plt

plt.bar(results.keys(), results.values())
plt.ylabel("Doğruluk (%)")
plt.title("Farklı Test Setlerinde Model Performansı")
plt.show()
