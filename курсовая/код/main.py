from model import SigNet, ContrastiveLoss
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import pandas as pd
import torch.optim as optim
from torchvision import transforms
from metrics import accuracy
from argparse import ArgumentParser

# Использование сида для сохранения одной и то же случайности, и использования Cuda(GPU)
seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.is_available()
print('Device: {}'.format(device))

def train(model, optimizer, criterion, dataloader, log_interval=50):
    # Изменения режима модели
    model.train()
    running_loss = 0
    number_samples = 0

    for batch_idx, (x1, x2, y) in enumerate(tqdm(dataloader, desc="Training")):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        optimizer.zero_grad()
        x1, x2 = model(x1, x2)
        loss = criterion(x1, x2, y)
        loss.backward()
        optimizer.step()

        number_samples += len(x1)
        running_loss += loss.item() * len(x1)
        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(dataloader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx+1, len(dataloader), running_loss / number_samples))
            running_loss = 0
            number_samples = 0

@torch.no_grad()
def eval(model, criterion, dataloader, log_interval=50):
    model.eval()
    running_loss = 0
    number_samples = 0

    distances = []

    for batch_idx, (x1, x2, y) in enumerate(tqdm(dataloader, desc="Evaluating")):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        x1, x2 = model(x1, x2)
        loss = criterion(x1, x2, y)
        distances.extend(zip(torch.pairwise_distance(x1, x2, 2).cpu().tolist(), y.cpu().tolist()))

        number_samples += len(x1)
        running_loss += loss.item() * len(x1)

        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(dataloader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx+1, len(dataloader), running_loss / number_samples))

    distances, y = zip(*distances)
    distances, y = torch.tensor(distances), torch.tensor(y)
    max_accuracy = accuracy(distances, y)
    print(f'Max accuracy: {max_accuracy}')
    return running_loss / number_samples, max_accuracy


# Класс для загрузки данных
class SignatureDataset(Dataset):
    def __init__(self, pairs_file, root_dir, transform=None):
        self.pairs = pd.read_csv(pairs_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]

        # Пути к изображениям
        img1_path = os.path.normpath(os.path.join(self.root_dir, row['path1']))
        img2_path = os.path.normpath(os.path.join(self.root_dir, row['path2']))
        label = torch.tensor(row['label'], dtype=torch.float32)

        # Загрузка изображений как тензоров, добавил перевод из double и float
        img1 = torch.load(img1_path, weights_only=True).to(torch.float32)
        img2 = torch.load(img2_path, weights_only=True).to(torch.float32)

        # Применение трансформации, если требуется
        if self.transform:
            img1 = self.transform(img1)  # Ожидается, что transform работает с тензорами
            img2 = self.transform(img2)

        return img1, img2, label

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--dataset', type=str, choices=['cedar', 'bengali', 'hindi'], default='cedar')
    args = parser.parse_args()
    print(args)

    model = SigNet().to(device)
    criterion = ContrastiveLoss(alpha=1, beta=1, margin=1).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-5, eps=1e-8, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.1)
    num_epochs = 20

    image_transform = transforms.Compose([])

    train_pairs_file = "UTSig_split/train_pairs.csv"
    test_pairs_file = "UTSig_split/test_pairs.csv"
    train_root_dir = "UTSig_split/train"
    test_root_dir = "UTSig_split/test"

    # Создание датасетов и загрузчиков данных
    train_dataset = SignatureDataset(train_pairs_file, train_root_dir, transform=image_transform)
    test_dataset = SignatureDataset(test_pairs_file, test_root_dir, transform=image_transform)

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.train()
    print(model)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('Training', '-' * 20)
        train(model, optimizer, criterion, trainloader)
        print('Evaluating', '-' * 20)
        loss, acc = eval(model, criterion, testloader)
        scheduler.step()

        to_save = {
            'model': model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optim': optimizer.state_dict(),
        }

        print('Saving checkpoint..')
        torch.save(to_save, 'checkpoints/epoch_{}_loss_{:.3f}_acc_{:.3f}.pt'.format(epoch, loss, acc))

    print('Done')