# Лабораторная практическая работа 3 по дисциплине МРЗВИС
# Выполнена студентами группы 221701 БГУИР Дичковским Владимиром Андреевичем
# 08.12.2025
# Использованные материалы:
# Deep Equilibrium Models S. Bai, J.Z. Kolter, and V. Koltun.
# Advances in Neural Information Processing Systems (NeurIPS) 2019

import torch
import torch.nn as nn
from deq import DEQFixedPoint
from resnet_layer import ResNetLayer


class DEQSequenceModel(nn.Module):
    """
    DEQ-модель, которая ищет фиксированную точку z = f(z, x),
    а затем декодирует найденное состояние в выходное значение.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, activation):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Инициализируем слой f(z, x)
        self.f_layer = ResNetLayer(hidden_dim, input_dim, activation)

        # Выходной декодер (линейный слой), преобразующий скрытое состояние z* в предсказание
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        bsz = x.shape[0]
        # Начальное приближение z_init = 0
        z_init = torch.zeros(bsz, self.hidden_dim, device=x.device)

        # DEQFixedPoint ищет z*, такое что z* = f(z*, x)
        # Используется кастомный autograd function для неявного дифференцирования
        z_star = DEQFixedPoint.apply(self.f_layer, x, z_init)

        return self.decoder(z_star)