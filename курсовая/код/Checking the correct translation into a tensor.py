import torch
import matplotlib.pyplot as plt

# Путь к вашему тензору
tensor_path = "UTSig_preprocessed_tensors/Forgery/forgeries_1_1.pt"

# Загрузка тензора
tensor = torch.load(tensor_path)

# Проверка формы тензора
print("Форма тензора:", tensor.shape)

# Определение ширины и высоты
if len(tensor.shape) == 3:  # Формат (C, H, W)
    channels, height, width = tensor.shape
    print(f"Каналы: {channels}, Высота: {height}, Ширина: {width}")
elif len(tensor.shape) == 2:  # Формат (H, W) для Grayscale
    height, width = tensor.shape
    print(f"Высота: {height}, Ширина: {width}")
else:
    print("Тензор имеет некорректную форму.")

# Визуализация, если это изображение
if len(tensor.shape) == 3:
    image = tensor.permute(1, 2, 0).numpy()  # Преобразование в (H, W, C)
    plt.imshow(image)
    plt.axis('off')
    plt.show()