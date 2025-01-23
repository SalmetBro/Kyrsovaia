import os
import csv
from collections import defaultdict

# Путь к разделенным данным
base_dir = "UTSig_split/"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

def parse_filename(filename):
    """
    Парсит имя файла и возвращает идентификатор автора и номер подписи.
    Например, Forgeries_1_2 -> (1, 2)
    """
    parts = filename.replace(".pt", "").split("_")
    return int(parts[1]), int(parts[2])

def create_pairs(data_dir):
    """
    Создает пары для указанной директории (train/test).
    Возвращает список пар вида [(path1, path2, label)], где:
      - path1, path2: пути к изображениям,
      - label: 1, если пара - подлинник-подлинник; 0, если подделка-подлинник.
    """
    genuine_dir = os.path.join(data_dir, "Genuine")
    forgery_dir = os.path.join(data_dir, "Forgery")

    # Словари для хранения подписей по авторам
    genuine_signatures = defaultdict(list)
    forgery_signatures = defaultdict(list)

    # Группируем подписи по авторам
    for filename in os.listdir(genuine_dir):
        author_id, signature_id = parse_filename(filename)
        genuine_signatures[author_id].append(os.path.join("Genuine", filename))

    for filename in os.listdir(forgery_dir):
        author_id, signature_id = parse_filename(filename)
        forgery_signatures[author_id].append(os.path.join("Forgery", filename))

    pairs = []

    # Создаем пары подлинник-подлинник
    for author_id, signatures in genuine_signatures.items():
        for i in range(len(signatures)):
            for j in range(i + 1, len(signatures)):
                pairs.append((signatures[i], signatures[j], 1))  # label = 1 (подлинник-подлинник)

    # Создаем пары подделка-подлинник
    for author_id, forgeries in forgery_signatures.items():
        if author_id in genuine_signatures:
            genuine = genuine_signatures[author_id]
            for forgery in forgeries:
                for original in genuine:
                    pairs.append((forgery, original, 0))  # label = 0 (подделка-подлинник)

    return pairs

def save_pairs_to_csv(pairs, output_path):
    """
    Сохраняет пары в CSV файл.
    """
    with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["path1", "path2", "label"])  # Заголовок
        writer.writerows(pairs)

# Генерируем пары для обучающей и тестовой выборок
train_pairs = create_pairs(train_dir)
test_pairs = create_pairs(test_dir)

# Сохраняем пары в CSV файлы
train_csv_path = os.path.join(base_dir, "train_pairs.csv")
test_csv_path = os.path.join(base_dir, "test_pairs.csv")

save_pairs_to_csv(train_pairs, train_csv_path)
save_pairs_to_csv(test_pairs, test_csv_path)

print(f"Пары для обучения сохранены в: {train_csv_path}")
print(f"Пары для тестирования сохранены в: {test_csv_path}")
