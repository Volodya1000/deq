# Лабораторная практическая работа 3 по дисциплине МРЗВИС
# Выполнена студентами группы 221701 БГУИР Дичковским Владимиром Андреевичем
# 08.12.2025
# Использованные материалы:
# Deep Equilibrium Models S. Bai, J.Z. Kolter, and V. Koltun.
# Advances in Neural Information Processing Systems (NeurIPS) 2019

import torch
import torch.nn as nn
import torch.optim as optim
from models import DEQSequenceModel
from data import create_dataset

def train_and_evaluate(name, sequence, window_size=5, hidden_dim=40, epochs=150):
    """Обучение модели и возврат предсказаний"""
    print(f"\n--- Обучение на последовательности: {name} ---")

    if len(sequence) <= window_size + 1:
        print("Ошибка: Слишком короткая последовательность.")
        return None

    # Подготовка данных
    X, Y, min_val, scale = create_dataset(sequence, window_size)
    model = DEQSequenceModel(window_size, hidden_dim, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Обучение
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % (epochs // 5) == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

    # Предсказание
    model.eval()
    with torch.no_grad():
        pred_norm = model(X).numpy()

    # Денормализация
    pred_real = pred_norm * scale + min_val
    target_real = Y.numpy() * scale + min_val

    return target_real, pred_real, losses