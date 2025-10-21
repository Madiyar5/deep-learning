import math

# Исходные данные
pixels = [0.5, 0.8, 0.2]  # x1, x2, x3
weights = [0.3, -0.5, 0.7]  # w1, w2, w3 (будут меняться!)
bias = 0.5  # Начнем с 0.5 для гарантированного обучения
target = 0.45
learning_rate = 0.01  # Темп обучения


# --- Вспомогательные функции ---

def forward_pass(inputs, weights, bias):
    """Вычисляет взвешенную сумму (z) и активацию (a)"""
    z = 0
    for i in range(len(inputs)):
        z += weights[i] * inputs[i]
    z += bias

    # ReLU
    a = max(0, z)
    return z, a


def calculate_loss(target, prediction):
    """Функция потерь MSE"""
    return 0.5 * (target - prediction) ** 2


# --- Цикл Обучения ---

num_epochs = 100
print(f"Начальные веса: {weights}, Bias: {bias}, LR: {learning_rate}\n")

for epoch in range(1, num_epochs + 1):
    # 1. Прямой проход (FORWARD PASS)
    z, result = forward_pass(pixels, weights, bias)
    loss = calculate_loss(target, result)

    # 2. Обратный проход (BACKWARD PASS / Расчет градиентов)

    # Градиент ошибки по выходу (d_Loss / d_a)
    d_loss_d_a = result - target

    # Градиент активации по взвешенной сумме (d_a / d_z)
    if z > 0:
        d_a_d_z = 1.0
    else:
        # Если z <= 0, градиент ReLU = 0, и обучение прекращается.
        d_a_d_z = 0.0

        # Цепное правило: d_Loss / d_z
    d_loss_d_z = d_loss_d_a * d_a_d_z

    # Градиенты по параметрам (d_Loss / d_w_i и d_Loss / d_b)
    grad_bias = d_loss_d_z * 1.0  # d_z / d_b = 1

    grad_weights = []
    for x_i in pixels:
        grad_w_i = d_loss_d_z * x_i  # d_z / d_w_i = x_i
        grad_weights.append(grad_w_i)

    # 3. Обновление весов (GRADIENT DESCENT)

    # Обновление bias
    bias = bias - learning_rate * grad_bias

    # Обновление weights
    new_weights = []
    for w_i, grad_w_i in zip(weights, grad_weights):
        w_new = w_i - learning_rate * grad_w_i
        new_weights.append(w_new)
    weights = new_weights

    # 4. Вывод результата
    if epoch % 10 == 0 or epoch == 1:
        print(f"Эпоха {epoch:3d} | Выход: {result:.4f} | Loss: {loss:.6f} | w1: {weights[0]:.4f}")

# --- Финальный результат ---
final_z, final_result = forward_pass(pixels, weights, bias)
final_loss = calculate_loss(target, final_result)

print("\n" + "=" * 40)
print(f"Целевое значение (Target): {target}")
print(f"Финальный выход нейрона: {final_result:.4f}")
print(f"Финальный Loss: {final_loss:.6f}")
print(f"Финальные веса: {weights}")
print(f"Финальный Bias: {bias}")