import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np

# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Используется устройство: {device}')

# Гиперпараметры
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

# Загрузка и подготовка данных MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f'Размер обучающей выборки: {len(train_dataset)}')
print(f'Размер тестовой выборки: {len(test_dataset)}')


# ============= CNN Модель =============
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Первый сверточный блок
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Второй сверточный блок
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Полносвязные слои
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# ============= MLP Модель =============
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


# Функция для подсчета параметров
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Функция обучения
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# Функция тестирования
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return accuracy


# ============= Обучение CNN =============
print('\n' + '=' * 50)
print('ОБУЧЕНИЕ CNN МОДЕЛИ')
print('=' * 50)

cnn_model = CNNModel().to(device)
cnn_params = count_parameters(cnn_model)
print(f'Количество параметров CNN: {cnn_params:,}')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)

cnn_start_time = time.time()

for epoch in range(EPOCHS):
    train_loss, train_acc = train_model(cnn_model, train_loader, criterion, optimizer, device)
    print(f'Эпоха [{epoch + 1}/{EPOCHS}] - Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

cnn_training_time = time.time() - cnn_start_time
cnn_test_accuracy = test_model(cnn_model, test_loader, device)

print(f'\nВремя обучения CNN: {cnn_training_time:.2f} секунд')
print(f'Точность CNN на тесте: {cnn_test_accuracy:.2f}%')

# ============= Обучение MLP =============
print('\n' + '=' * 50)
print('ОБУЧЕНИЕ MLP МОДЕЛИ')
print('=' * 50)

mlp_model = MLPModel().to(device)
mlp_params = count_parameters(mlp_model)
print(f'Количество параметров MLP: {mlp_params:,}')

optimizer = optim.Adam(mlp_model.parameters(), lr=LEARNING_RATE)

mlp_start_time = time.time()

for epoch in range(EPOCHS):
    train_loss, train_acc = train_model(mlp_model, train_loader, criterion, optimizer, device)
    print(f'Эпоха [{epoch + 1}/{EPOCHS}] - Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

mlp_training_time = time.time() - mlp_start_time
mlp_test_accuracy = test_model(mlp_model, test_loader, device)

print(f'\nВремя обучения MLP: {mlp_training_time:.2f} секунд')
print(f'Точность MLP на тесте: {mlp_test_accuracy:.2f}%')

# ============= ИТОГОВОЕ СРАВНЕНИЕ =============
print('\n' + '=' * 50)
print('ИТОГОВОЕ СРАВНЕНИЕ')
print('=' * 50)
print(f'{"Модель":<10} {"Точность":>15} {"Параметры":>18} {"Время обучения":>20}')
print('-' * 70)
print(f'{"CNN":<10} {cnn_test_accuracy:>14.2f}% {cnn_params:>17,} {cnn_training_time:>16.2f} сек')
print(f'{"MLP":<10} {mlp_test_accuracy:>14.2f}% {mlp_params:>17,} {mlp_training_time:>16.2f} сек')
print('=' * 70)

# Вычисление преимуществ CNN
param_reduction = ((mlp_params - cnn_params) / mlp_params) * 100
acc_improvement = cnn_test_accuracy - mlp_test_accuracy

print(f'\nПреимущества CNN:')
print(f'  - Сокращение параметров: {param_reduction:.1f}%')
print(f'  - Улучшение точности: {acc_improvement:+.2f}%')
print(f'  - Соотношение времени: CNN/MLP = {cnn_training_time / mlp_training_time:.2f}')

# Сохранение моделей
torch.save(cnn_model.state_dict(), 'cnn_model.pth')
torch.save(mlp_model.state_dict(), 'mlp_model.pth')
print('\nМодели сохранены: cnn_model.pth, mlp_model.pth')