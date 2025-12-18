# Лабораторная практическая работа 3 по дисциплине МРЗВИС
# Выполнена студентами группы 221701 БГУИР Дичковским Владимиром Андреевичем
# 08.12.2025
# Использованные материалы:
# Deep Equilibrium Models S. Bai, J.Z. Kolter, and V. Koltun.
# Advances in Neural Information Processing Systems (NeurIPS) 2019

import torch
import torch.nn as nn
import torch.optim as optim
import logging
from deq_model import DEQSequenceModel
from data import create_dataset

logger = logging.getLogger(__name__)


def train_and_evaluate(seq_name, sequence, activation_func, act_name,
                       normalize, window_size=5, hidden_dim=40,
                       epochs=150):
    norm_status = "ON" if normalize else "OFF"
    logger.info(f"START: Seq='{seq_name}' | Act='{act_name}' | Norm={norm_status}")

    if len(sequence) <= window_size + 1:
        logger.error(f"FAIL: Sequence '{seq_name}' too short.")
        return None

    X, Y, min_val, scale = create_dataset(sequence, window_size, normalize=normalize)

    model = DEQSequenceModel(window_size, hidden_dim, 1, activation=activation_func)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Прямой проход (включает поиск неподвижной точки)
        out = model(X)
        loss = criterion(out, Y)

        # Обратный проход (через теорему о неявной функции)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Логируем каждые 20% эпох
        if epoch % (epochs // 5) == 0 or epoch == epochs - 1:
            logger.info(f"  [Epoch {epoch}] Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        pred_raw = model(X).numpy()

    pred_real = pred_raw * scale + min_val
    target_real = Y.numpy() * scale + min_val

    final_loss = losses[-1]
    logger.info(f"FINISH: Seq='{seq_name}' | Act='{act_name}' | Final Loss={final_loss:.6f}\n")

    return target_real, pred_real, losses