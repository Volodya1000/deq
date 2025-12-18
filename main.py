# Лабораторная практическая работа 3 по дисциплине МРЗВИС
# Выполнена студентами группы 221701 БГУИР Дичковским Владимиром Андреевичем
# 08.12.2025
# Использованные материалы:
# Deep Equilibrium Models S. Bai, J.Z. Kolter, and V. Koltun.
# Advances in Neural Information Processing Systems (NeurIPS) 2019

import matplotlib.pyplot as plt
import torch.nn as nn
import logging
import sys
from data import get_sequences
from train import train_and_evaluate

# Настройка логирования: писать и в файл, и в консоль
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("experiment_log.txt", mode='w'),  # Запись в файл
        logging.StreamHandler(sys.stdout)  # Вывод в консоль
    ]
)


def main():
    sequences = get_sequences()

    # Словарь функций активации для сравнения
    # ELU полезна для предотвращения затухания градиентов
    # Tanh стандартна для рекуррентных сетей
    activations = {
        "Tanh": nn.Tanh(),
        "ELU": nn.ELU(alpha=1.0)
    }

    # Конфигурации для тестирования нормализации
    # Преподаватель просил проверить работу без нормализации
    normalization_modes = [True, False]

    # Создаем графики
    # Будем строить по графику для каждой последовательности
    fig, axes = plt.subplots(len(sequences), 1, figsize=(10, 15))
    if len(sequences) == 1: axes = [axes]  # Корректировка если всего 1 график

    for i, (seq_name, seq) in enumerate(sequences.items()):
        ax = axes[i]

        # Параметры обучения зависят от длины последовательности
        w_size = 3 if len(seq) < 30 else 10
        h_dim = 40
        n_epochs = 300 if len(seq) < 30 else 100

        target_plotted = False

        logging.info(f"=== Processing Sequence: {seq_name} ===")

        # Перебор режимов нормализации
        for norm in normalization_modes:
            # Перебор функций активации
            for act_name, act_fn in activations.items():

                # Запуск обучения
                result = train_and_evaluate(
                    seq_name, seq, act_fn, act_name,
                    normalize=norm,
                    window_size=w_size,
                    hidden_dim=h_dim,
                    epochs=n_epochs
                )

                if result:
                    target, pred, _ = result

                    # Отрисовка целевого графика (один раз)
                    if not target_plotted:
                        ax.plot(target, 'k-', label='Target (Real)', linewidth=2.5, alpha=0.3)
                        target_plotted = True

                    # Стиль линии в зависимости от параметров
                    # Сплошная - с нормализацией, Пунктир - без
                    # Красный - Tanh, Синий - ELU
                    line_style = '-' if norm else '--'
                    color = 'r' if act_name == "Tanh" else 'b'
                    norm_label = "Norm" if norm else "NoNorm"

                    ax.plot(pred, color=color, linestyle=line_style,
                            label=f'{act_name} + {norm_label}', linewidth=1.5)

        ax.set_title(f"Sequence: {seq_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    logging.info("All experiments finished.")


if __name__ == "__main__":
    main()