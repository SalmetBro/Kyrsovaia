import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import torch
from torchvision import transforms

## Параметры предобработки
image_size = (155, 220)  # Размер изображений
to_tensor = transforms.ToTensor()  # Преобразование в тензор

# Функция для удаления фона
def remove_background(image: Image.Image) -> Image.Image:
    # Конвертируем изображение в массив numpy
    image_array = np.array(image)

    # Определяем фон как пиксели с интенсивностью выше порога
    threshold = 240  # Порог для белого фона
    mask = image_array < threshold

    # Устанавливаем фон в белый цвет (255), остальные значения оставляем
    image_array[~mask] = 255
    return Image.fromarray(image_array)

# Функция для обработки одного изображения c изменением размеров
def preprocess_image(image_path: str, image_size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(image_path)

    # Приведение к градациям серого
    image = ImageOps.grayscale(image)

    # Увеличение контраста
    image = ImageEnhance.Contrast(image).enhance(2)

    # Убираем шум и подчеркиваем важные элементы
    image = image.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=1))

    # Изменение размера
    image = image.resize(image_size, Image.Resampling.LANCZOS)

    # Преобразуем изображение в массив и нормализуем изображение
    image = np.array(image)
    image = image / 255.0

    # Преобразование в тензор
    image_tensor = to_tensor(image)
    return image_tensor

# Основной код
def preprocess_dataset_to_tensors(dataset_path: str, output_path: str):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for folder in ['Genuine', 'Forgery']:
        input_folder = os.path.join(dataset_path, folder)
        output_folder = os.path.join(output_path, folder)
        os.makedirs(output_folder, exist_ok=True)

        for image_name in os.listdir(input_folder):
            input_image_path = os.path.join(input_folder, image_name)
            output_tensor_path = os.path.join(output_folder, image_name.replace('.png', '.pt'))

            if image_name.lower().endswith('.png'):
                processed_tensor = preprocess_image(input_image_path, image_size)
                # Сохранение тензора
                torch.save(processed_tensor, output_tensor_path)

# Визуализация первых 5 изображений для проверки
def visualize_tensors(dataset_path: str, folder: str, n: int = 5):
    folder_path = os.path.join(dataset_path, folder)
    tensor_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')][:n]

    for i, tensor_file in enumerate(tensor_files):
        tensor_path = os.path.join(folder_path, tensor_file)
        image_tensor = torch.load(tensor_path)
        image = transforms.ToPILImage()(image_tensor)

        plt.subplot(1, n, i + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.title(tensor_file)
    plt.show()

# Путь к датасету
input_dataset_path = "UTSig"  # Укажите путь к оригинальному датасету
output_dataset_path = "UTSig_preprocessed_tensors"  # Укажите путь для сохранения обработанных данных

# Предобработка изображений
preprocess_dataset_to_tensors(input_dataset_path, output_dataset_path)

# Визуализация
print("Подлинные подписи после обработки:")
visualize_tensors(output_dataset_path, 'Genuine')

print("Поддельные подписи после обработки:")
visualize_tensors(output_dataset_path, 'Forgery')