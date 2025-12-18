# Лабораторная практическая работа 3 по дисциплине МРЗВИС
# Выполнена студентами группы 221701 БГУИР Дичковским Владимиром Андреевичем
# 08.12.2025
# Использованные материалы:
# Deep Equilibrium Models S. Bai, J.Z. Kolter, and V. Koltun.
# Advances in Neural Information Processing Systems (NeurIPS) 2019

import torch
import torch.nn as nn

class ResNetLayer(nn.Module):
    """
    Один слой ResNet для DEQ.
    Отвечает за вычисление функции f(z, x), для которой ищем неподвижную точку.
    """

    def __init__(self, n_hidden, n_input, activation):
        super().__init__()
        # Линейный слой для входа x (внешнее воздействие)
        self.fc_x = nn.Linear(n_input, n_hidden, bias=False)
        # Линейный слой для скрытого состояния z (предыдущее приближение)
        self.fc_z = nn.Linear(n_hidden, n_hidden)
        self.activation = activation

    def forward(self, z, x):
        # Основная формула: f(z, x) = activation(Wz * z + Wx * x)
        # ищем такое z, чтобы z = f(z, x)
        return self.activation(self.fc_z(z) + self.fc_x(x))