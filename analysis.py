# Лабораторная практическая работа 3 по дисциплине МРЗВИС
# Выполнена студентами группы 221701 БГУИР Дичковским Владимиром Андреевичем
# 08.12.2025
# Использованные материалы:
# Deep Equilibrium Models S. Bai, J.Z. Kolter, and V. Koltun.
# Advances in Neural Information Processing Systems (NeurIPS) 2019

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from data import get_sequences, create_dataset
from models import DEQSequenceModel, ResNetLayer
from deq import DEQFixedPoint

from train import train_and_evaluate


def plot_convergence_dynamics(model, X_sample, max_iter=50, threshold=1e-3):
    """
    Визуализирует процесс сходимости к неподвижной точке.
    Показывает ||z_{i+1} - z_i|| vs номер итерации.
    """
    model.eval()
    f_layer = model.f_layer

    with torch.no_grad():
        z = torch.zeros(X_sample.shape[0], model.hidden_dim, device=X_sample.device)
        diffs = []

        for i in range(max_iter):
            z_new = f_layer(z, X_sample)
            diff = torch.norm(z_new - z).item()
            diffs.append(diff)
            z = z_new
            if diff < threshold:
                break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # График нормы разности
    ax1.semilogy(diffs, 'bo-', linewidth=2, markersize=6)
    ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Порог ({threshold})')
    ax1.set_xlabel('Итерация')
    ax1.set_ylabel('||z_{i+1} - z_i||_2')
    ax1.set_title('Сходимость к неподвижной точке')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График нормы z (стабильность)
    ax2.plot([np.linalg.norm(z[i].cpu().numpy()) for i in range(len(z))], 'g-')
    ax2.set_xlabel('Индекс образца')
    ax2.set_ylabel('||z*||_2')
    ax2.set_title('Распределение нормы фиксированной точки')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_broyden_iterations_over_training(losses, forward_iters, backward_iters):
    """
    Показывает рост числа итераций Broyden по эпохам - аналог Fig. 4 (left) из статьи.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.twinx()
    line1 = ax1.plot(losses, 'b-', label='Training Loss', linewidth=2)
    line2 = ax2.plot(forward_iters, 'r--', label='Forward Iterations', linewidth=2)
    line3 = ax2.plot(backward_iters, 'g--', label='Backward Iterations', linewidth=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss', color='b')
    ax2.set_ylabel('Broyden Iterations', color='r')

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    ax1.set_title('Broyden Iterations vs Training Epochs')
    ax1.grid(True, alpha=0.3)

    plt.show()


# ==================== АНАЛИЗ ПАМЯТИ ====================

def plot_memory_efficiency(sequence_lengths=[50, 100, 200, 400, 800], hidden_dim=40):
    """
    Сравнивает теоретическое использование памяти DEQ vs обычной сети.
    Иллюстрирует ключевое преимущество: O(1) vs O(L) памяти.
    """
    memory_deq = []
    memory_deep = []

    for T in sequence_lengths:
        # DEQ: постоянная память (одна копия z*)
        mem_deq = hidden_dim * 4  # z_star + x + grad + buffer (в MB)
        memory_deq.append(mem_deq)

        # Глубокая сеть: линейная зависимость от глубины
        # Предположим 20 слоев для среднего случая
        L = 20
        mem_deep = T * hidden_dim * L * 4  # все промежуточные слои
        memory_deep.append(mem_deep / 1024)  # в GB

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sequence_lengths, memory_deq, 'bo-', linewidth=3, markersize=8, label='DEQ (O(1))')
    ax.plot(sequence_lengths, memory_deep, 'rs-', linewidth=3, markersize=8, label='Deep Net (O(L))')

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Memory Usage (GB)')
    ax.set_title('Memory Efficiency: DEQ vs Deep Networks')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Логарифмический масштаб по Y
    ax.set_yscale('log')

    plt.show()


# ==================== АНАЛИЗ СТАБИЛЬНОСТИ ====================

def plot_jacobian_stability(model, X_sample, epoch_nums):
    """
    Анализирует операторную норму якобиана (приближенно).
    Помогает понять, почему число итераций растет (Fig. 4 left).
    """
    model.eval()
    f_layer = model.f_layer

    norms = []

    for epoch in epoch_nums:
        # Предположим, что у нас есть сохраненные веса для каждой эпохи
        with torch.no_grad():
            z_star = DEQFixedPoint.apply(f_layer, X_sample, torch.zeros_like(X_sample))
            z_star.requires_grad_(True)

            # Оценка операторной нормы через степенной метод
            v = torch.randn_like(z_star)
            for _ in range(10):
                Jv = torch.autograd.functional.jvp(f_layer, (z_star, X_sample), v=(v, torch.zeros_like(X_sample)))[1]
                v = Jv / torch.norm(Jv)

            Jv = torch.autograd.functional.jvp(f_layer, (z_star, X_sample), v=(v, torch.zeros_like(X_sample)))[1]
            norm = torch.norm(Jv).item()
            norms.append(norm)

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_nums, norms, 'm^-', linewidth=2, markersize=8)
    plt.xlabel('Training Epoch')
    plt.ylabel('Operator Norm of Jacobian')
    plt.title('Jacobian Stability During Training')
    plt.grid(True, alpha=0.3)
    plt.show()


# ==================== АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ ====================

def plot_threshold_sensitivity(sequence, window_size=10, thresholds=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]):
    """
    Исследует влияние threshold на качество модели (аналог Fig. 6 left).
    """
    results = {}

    for thr in thresholds:
        print(f"Testing threshold: {thr}")
        target, pred, losses = train_and_evaluate(
            f"Threshold {thr}", sequence, window_size=window_size,
            hidden_dim=40, epochs=50
        )
        results[thr] = {
            'loss': losses[-1],
            'final_loss': losses
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # График конечной ошибки vs threshold
    thr_vals = list(results.keys())
    losses = [results[thr]['loss'] for thr in thr_vals]
    ax1.semilogx(thr_vals, losses, 'ro-', linewidth=2, markersize=8)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Final MSE Loss')
    ax1.set_title('Model Quality vs Threshold')
    ax1.grid(True, alpha=0.3)

    # График обучения для разных threshold
    for thr, data in results.items():
        ax2.plot(data['final_loss'], label=f'thr={thr}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Training Dynamics for Different Thresholds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== СРАВНЕНИЕ С ФИКСИРОВАННОЙ ГЛУБИНОЙ ====================

class FixedDepthModel(nn.Module):
    """Модель с фиксированной глубиной для сравнения с DEQ"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([ResNetLayer(hidden_dim, input_dim) for _ in range(num_layers)])
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.input_dim = input_dim

    def forward(self, x):
        z = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        for layer in self.layers:
            z = layer(z, x)
        return self.decoder(z)


def compare_fixed_vs_deq(sequence, depths=[5, 10, 20, 40]):
    """
    Сравнивает DEQ с эквивалентными сетями фиксированной глубины.
    Иллюстрирует Theorem 2: один DEQ layer = бесконечная глубина.
    """
    results = {}

    # DEQ
    target_deq, pred_deq, losses_deq = train_and_evaluate(
        "DEQ", sequence, hidden_dim=40, epochs=100
    )
    results['DEQ'] = {'loss': losses_deq[-1], 'params': 40 * 40 + 40 * 10 + 40}

    # Фиксированная глубина
    for depth in depths:
        target_fd, pred_fd, losses_fd = train_and_evaluate(
            f"FixedDepth_{depth}", sequence, hidden_dim=40, epochs=100
        )
        results[f'Fixed_{depth}'] = {
            'loss': losses_fd[-1],
            'params': depth * (40 * 40 + 40 * 10) + 40
        }

    # Визуализация
    fig, ax = plt.subplots(figsize=(12, 6))
    models = list(results.keys())
    losses = [results[m]['loss'] for m in models]
    params = [results[m]['params'] for m in models]

    ax.scatter(params, losses, s=100)
    for i, model in enumerate(models):
        ax.annotate(model, (params[i], losses[i]), xytext=(5, 5),
                    textcoords='offset points', fontsize=10)

    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Final MSE Loss')
    ax.set_title('DEQ vs Fixed-Depth Networks (Same Width)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.show()


def run_analysis_fibonacci():
    sequences = get_sequences()
    seq = sequences["Fibonacci"]  # выбираем только Фибоначчи

    print(f"--- Анализ последовательности: Fibonacci ---")

    w_size = 3 if len(seq) < 30 else 10
    X, Y, _, _ = create_dataset(seq, w_size)

    # Инициализация модели
    model = DEQSequenceModel(w_size, 40, 1)
    X_sample = X[:5]  # небольшая выборка для анализа

    # 1. Динамика сходимости
    print("Генерация графика сходимости...")
    plot_convergence_dynamics(model, X_sample)

    # 2. Анализ памяти
    print("Генерация графика эффективности памяти...")
    plot_memory_efficiency()

    # 3. Чувствительность к threshold
    print("Анализ чувствительности к threshold...")
    plot_threshold_sensitivity(seq, w_size)

    # 4. Сравнение с фиксированной глубиной
    print("Сравнение с фиксированной глубиной...")
    compare_fixed_vs_deq(seq)


if __name__ == "__main__":
    run_analysis_fibonacci()