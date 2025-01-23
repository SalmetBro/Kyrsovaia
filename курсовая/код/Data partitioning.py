import os
import shutil
from sklearn.model_selection import train_test_split

# Копируем изображения в соответствующие папки
def copy_images(image_list, label_list, base_dir, root_dir):
    for img_path, label in zip(image_list, label_list):
        if label == 1:  # Genuine
            dest_dir = os.path.join(base_dir, "Genuine")
        else:  # Forgery
            dest_dir = os.path.join(base_dir, "Forgery")
        os.makedirs(dest_dir, exist_ok=True)  # Создаем папки, если их нет
        abs_img_path = os.path.join(root_dir, img_path)  # Преобразуем в абсолютный путь
        shutil.copy(abs_img_path, dest_dir)

# Путь к данным
dataset_dir = "UTSig_preprocessed_tensors/"
genuine_dir = os.path.join(dataset_dir, "Genuine")
forgery_dir = os.path.join(dataset_dir, "Forgery")

# Список всех изображений
genuine_images = [os.path.join("Genuine", img) for img in os.listdir(genuine_dir) if img.endswith('.pt')]
forgery_images = [os.path.join("Forgery", img) for img in os.listdir(forgery_dir) if img.endswith('.pt')]

# Создаем метки (1 для подлинных, 0 для поддельных)
genuine_labels = [1] * len(genuine_images)
forgery_labels = [0] * len(forgery_images)

# Объединяем изображения и метки
all_images = genuine_images + forgery_images
all_labels = genuine_labels + forgery_labels

# Разделяем данные на обучающую и тестовую выборки
train_images, test_images, train_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

# Создаем папки для разделенных данных
output_dir = "UTSig_split/"
for subdir in ["train/Genuine", "train/Forgery", "test/Genuine", "test/Forgery"]:
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

# Копируем обучающие и тестовые данные
copy_images(train_images, train_labels, os.path.join(output_dir, "train"), dataset_dir)
copy_images(test_images, test_labels, os.path.join(output_dir, "test"), dataset_dir)

print("Данные успешно разделены и сохранены!")
