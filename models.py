# Лабораторная практическая работа 3 по дисциплине МРЗВИС
# Выполнена студентами группы 221701 БГУИР Дичковским Владимиром Андреевичем
# 08.12.2025
# Использованные материалы:
# Deep Equilibrium Models S. Bai, J.Z. Kolter, and V. Koltun.
# Advances in Neural Information Processing Systems (NeurIPS) 2019

import torch
import torch.nn as nn
from deq import DEQFixedPoint


class ResNetLayer(nn.Module):
    """Один слой ResNet для DEQ"""

    def __init__(self, n_hidden, n_input):
        super().__init__()
        self.fc_x = nn.Linear(n_input, n_hidden, bias=False)
        self.fc_z = nn.Linear(n_hidden, n_hidden)
        self.activation = nn.Tanh()

    def forward(self, z, x):
        # Основная формула: f(z, x) = tanh(Wz * z + Wx * x)
        return self.activation(self.fc_z(z) + self.fc_x(x))


class DEQSequenceModel(nn.Module):
    """DEQ-модель, которая ищет фиксированную точку z = f(z, x),
    а затем декодирует найденное состояние."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.f_layer = ResNetLayer(hidden_dim, input_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        bsz = x.shape[0]
        z_init = torch.zeros(bsz, self.hidden_dim, device=x.device)
        # DEQ ищет z*, такое что z* = f(z*, x)
        z_star = DEQFixedPoint.apply(self.f_layer, x, z_init)
        return self.decoder(z_star)