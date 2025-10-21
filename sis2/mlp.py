"""
MLP с нуля - учимся понимать, как работает нейросеть
Шаг за шагом от простого к сложному
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("ЧАСТЬ 1: ОДИН НЕЙРОН")
print("=" * 60)

# Создадим простое изображение 3x3 (представим, что это очень маленькая картинка)
image = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

print("Наше 'изображение' 3x3:")
print(image)

# Превращаем в плоский вектор (это делает MLP!)
flat_image = image.flatten()
print(f"\nПосле flatten: {flat_image}")
print(f"Размер: {flat_image.shape}")

# Создаем ОДИН нейрон
weights = np.random.randn(9)  # 9 случайных весов
bias = np.random.randn(1)

print(f"\nВеса нейрона: {weights}")
print(f"Смещение: {bias}")

# Вычисляем выход нейрона
output = np.dot(flat_image, weights) + bias
print(f"\nДо активации: {output}")

# Применяем ReLU
output_relu = np.maximum(0, output)
print(f"После ReLU: {output_relu}")

print("\n" + "=" * 60)
print("ЧАСТЬ 2: СЛОЙ НЕЙРОНОВ")
print("=" * 60)

# Теперь создадим СЛОЙ из 3 нейронов
num_neurons = 3
weights_layer = np.random.randn(9, num_neurons)  # 9 входов → 3 нейрона
bias_layer = np.random.randn(num_neurons)

print(f"\nМатрица весов: {weights_layer.shape}")
print(f"У каждого из {num_neurons} нейронов по 9 весов")

# Прогоняем через слой
layer_output = np.dot(flat_image, weights_layer) + bias_layer
layer_output = np.maximum(0, layer_output)  # ReLU

print(f"\nВыход слоя: {layer_output}")
print(f"Размер: {layer_output.shape}")

print("\n" + "=" * 60)
print("ЧАСТЬ 3: ПОЛНАЯ MLP СЕТЬ (2 слоя)")
print("=" * 60)

# Входные данные
input_size = 9
hidden_size = 5  # первый слой: 5 нейронов
output_size = 2  # второй слой: 2 нейрона (например, "кот" или "собака")

# Инициализируем веса
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros(hidden_size)

W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros(output_size)

print(f"Слой 1: {input_size} → {hidden_size}")
print(f"Слой 2: {hidden_size} → {output_size}")

# Forward pass (прямой проход)
print("\n--- FORWARD PASS ---")

# Слой 1
z1 = np.dot(flat_image, W1) + b1
a1 = np.maximum(0, z1)  # ReLU
print(f"После слоя 1: {a1}")

# Слой 2
z2 = np.dot(a1, W2) + b2
print(f"Финальный выход: {z2}")

# Softmax (превращаем в вероятности)
exp_scores = np.exp(z2)
probs = exp_scores / np.sum(exp_scores)
print(f"Вероятности: {probs}")
print(f"Класс 0: {probs[0] * 100:.1f}%, Класс 1: {probs[1] * 100:.1f}%")

print("\n" + "=" * 60)
print("ЧАСТЬ 4: ВИЗУАЛИЗАЦИЯ РАБОТЫ MLP")
print("=" * 60)

# Создадим простой пример с реальными данными
np.random.seed(42)

# Генерируем данные: 2 класса точек на плоскости
# Класс 0: точки вокруг (2, 2)
class_0 = np.random.randn(50, 2) + np.array([2, 2])
# Класс 1: точки вокруг (5, 5)
class_1 = np.random.randn(50, 2) + np.array([5, 5])

X = np.vstack([class_0, class_1])
y = np.array([0] * 50 + [1] * 50)

print(f"Данные: {X.shape}")
print(f"Метки: {y.shape}")

# Визуализация
plt.figure(figsize=(8, 6))
plt.scatter(class_0[:, 0], class_0[:, 1], c='blue', label='Класс 0', alpha=0.6)
plt.scatter(class_1[:, 0], class_1[:, 1], c='red', label='Класс 1', alpha=0.6)
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Наши данные для классификации')
plt.legend()
plt.grid(True)
plt.savefig('data_visualization.png', dpi=150, bbox_inches='tight')
print("\n✅ График сохранен: data_visualization.png")
plt.close()

print("\n" + "=" * 60)
print("ЧАСТЬ 5: ОБУЧЕНИЕ ПРОСТОЙ MLP")
print("=" * 60)

# Простая MLP для обучения
input_dim = 2
hidden_dim = 4
output_dim = 2
learning_rate = 0.01
num_iterations = 1000

# Инициализация весов
W1 = np.random.randn(input_dim, hidden_dim) * 0.1
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim) * 0.1
b2 = np.zeros(output_dim)

print(f"Архитектура: {input_dim} → {hidden_dim} → {output_dim}")
print(f"Параметров: {W1.size + b1.size + W2.size + b2.size}")

# Обучение
losses = []

for iteration in range(num_iterations):
    # Forward pass
    z1 = X.dot(W1) + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = a1.dot(W2) + b2

    # Softmax
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Loss (cross-entropy)
    correct_logprobs = -np.log(probs[range(len(y)), y])
    loss = np.sum(correct_logprobs) / len(y)
    losses.append(loss)

    if iteration % 100 == 0:
        # Точность
        predicted = np.argmax(probs, axis=1)
        accuracy = np.mean(predicted == y) * 100
        print(f"Итерация {iteration}: Loss = {loss:.4f}, Accuracy = {accuracy:.1f}%")

    # Backward pass (градиенты)
    dscores = probs
    dscores[range(len(y)), y] -= 1
    dscores /= len(y)

    dW2 = a1.T.dot(dscores)
    db2 = np.sum(dscores, axis=0)

    dhidden = dscores.dot(W2.T)
    dhidden[a1 <= 0] = 0  # ReLU gradient

    dW1 = X.T.dot(dhidden)
    db1 = np.sum(dhidden, axis=0)

    # Обновление весов
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# Финальная точность
z1 = X.dot(W1) + b1
a1 = np.maximum(0, z1)
z2 = a1.dot(W2) + b2
exp_scores = np.exp(z2)
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
predicted = np.argmax(probs, axis=1)
final_accuracy = np.mean(predicted == y) * 100

print(f"\n🎉 Финальная точность: {final_accuracy:.1f}%")

# График обучения
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Итерация')
plt.ylabel('Loss')
plt.title('Процесс обучения')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(class_0[:, 0], class_0[:, 1], c='blue', label='Класс 0', alpha=0.6)
plt.scatter(class_1[:, 0], class_1[:, 1], c='red', label='Класс 1', alpha=0.6)

# Решающая граница
h = 0.1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = np.c_[xx.ravel(), yy.ravel()]

z1 = Z.dot(W1) + b1
a1 = np.maximum(0, z1)
z2 = a1.dot(W2) + b2
Z = np.argmax(z2, axis=1)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Решающая граница')
plt.legend()

plt.tight_layout()
plt.savefig('mlp_training.png', dpi=150, bbox_inches='tight')
print("\n✅ График обучения сохранен: mlp_training.png")

print("\n" + "=" * 60)
print("ИТОГ: ЧТО МЫ УЗНАЛИ")
print("=" * 60)
print("""
1. Нейрон = взвешенная сумма входов + bias + активация
2. Слой = много нейронов параллельно
3. MLP = несколько слоев друг за другом
4. Forward pass = данные → слой1 → слой2 → выход
5. Обучение = подбор весов, чтобы минимизировать ошибку

MLP для изображений:
❌ Flatten разрушает структуру
❌ Много параметров (каждый пиксель ко всем нейронам)
❌ Не использует локальность (соседние пиксели связаны)

Теперь переходи к CNN, чтобы решить эти проблемы! 🚀
""")