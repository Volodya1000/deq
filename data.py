# Лабораторная практическая работа 3 по дисциплине МРЗВИС
# Выполнена студентами группы 221701 БГУИР Дичковским Владимиром Андреевичем
# 08.12.2025
# Использованные материалы:
# Deep Equilibrium Models S. Bai, J.Z. Kolter, and V. Koltun.
# Advances in Neural Information Processing Systems (NeurIPS) 2019


import numpy as np
import torch


def create_dataset(sequence, window_size):
    """Создание датасета для временных рядов с нормализацией"""
    data = np.array(sequence, dtype=np.float32)
    min_val = np.min(data)
    max_val = np.max(data)
    scale = max_val - min_val if max_val != min_val else 1.0

    data_norm = (data - min_val) / scale

    X, Y = [], []
    for i in range(len(data_norm) - window_size):
        X.append(data_norm[i:i + window_size])
        Y.append(data_norm[i + window_size])

    return torch.tensor(np.array(X)), torch.tensor(np.array(Y)).unsqueeze(1), min_val, scale


def get_sequences():
    """Генератор тестовых последовательностей"""
    seqs = {}

    # 1. Зашумленный синус
    t = np.linspace(0, 8 * np.pi, 200)
    seqs["Noisy Sine"] = np.sin(t) + 0.05 * np.random.normal(size=len(t))

    # 2. Фибоначчи
    fib = [1, 1]
    for i in range(18):
        fib.append(fib[-1] + fib[-2])
    seqs["Fibonacci"] = fib

    # 3. Затухающий косинус
    t2 = np.linspace(0, 15, 100)
    seqs["Damped Cos"] = np.exp(-t2 / 5) * np.cos(t2)

    np.random.seed(42)
    seqs["Random Binary"] = np.random.choice([0, 1], size=50)

    return seqs