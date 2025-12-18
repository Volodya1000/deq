# Лабораторная практическая работа 3 по дисциплине МРЗВИС
# Скрипт генерации изображений для статьи (Финальная версия)
# Выполнена студентами группы 221701 БГУИР Дичковским Владимиром Андреевичем

import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
import sys
from data import get_sequences
from train import train_and_evaluate

# === КОНФИГУРАЦИЯ ===
OUTPUT_DIR = os.path.join("images", "article_final")

# Настройка логгера
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def save_plot(fig, filename):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved graph: {path}")


def generate_images():
    sequences = get_sequences()

    activations = {
        "Tanh": nn.Tanh(),
        "ELU": nn.ELU(alpha=1.0)
    }

    results = {}
    logger.info("--- 1. ЗАПУСК ОБУЧЕНИЯ МОДЕЛЕЙ ---")

    for seq_name, seq in sequences.items():
        results[seq_name] = {}
        # Используем достаточно эпох для сходимости
        w_size = 3 if len(seq) < 30 else 10
        h_dim = 40
        n_epochs = 200

        for act_name, act_fn in activations.items():
            results[seq_name][act_name] = {}
            for norm in [True, False]:
                logger.info(f"Processing: {seq_name} | {act_name} | Norm={norm}")
                res = train_and_evaluate(seq_name, seq, act_fn, act_name, normalize=norm,
                                         window_size=w_size, hidden_dim=h_dim, epochs=n_epochs)
                if res:
                    results[seq_name][act_name][norm] = res

    logger.info("\n--- 2. ГЕНЕРАЦИЯ ГРАФИКОВ ---\n")
    plt.style.use('seaborn-v0_8-whitegrid')

    for seq_name in sequences.keys():
        safe_name = seq_name.replace(" ", "_")

        # --- ГРАФИК 1: ВЛИЯНИЕ НОРМАЛИЗАЦИИ НА TANH ---
        # (Исправлены подписи осей и заголовок)
        fig, ax = plt.subplots(figsize=(10, 6))

        target = results[seq_name]["Tanh"][True][0].flatten()
        pred_norm = results[seq_name]["Tanh"][True][1].flatten()
        pred_raw = results[seq_name]["Tanh"][False][1].flatten()

        # Для Фибоначчи делаем зум, иначе Tanh без нормы улетит в небеса и испортит масштаб
        if seq_name == "Fibonacci":
            max_val = max(target)
            ax.set_ylim(-max_val * 0.1, max_val * 1.5)

        ax.plot(target, 'k-', lw=3, alpha=0.3, label='Целевое значение (Target)')
        ax.plot(pred_norm, 'g--', lw=2, label='Tanh с нормализацией')
        ax.plot(pred_raw, 'r:', lw=2.5, label='Tanh без нормализации')

        ax.set_title(f"Влияние нормализации на Tanh: {seq_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Шаг последовательности (t)", fontsize=12)
        ax.set_ylabel("Значение $x_t$", fontsize=12)
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3)

        save_plot(fig, f"{safe_name}_1_Norm_Effect_Tanh.png")

        # --- ГРАФИК 2: СРАВНЕНИЕ ОШИБОК (ТОЧНОСТЬ) ---
        # (Добавлена подпись оси X, заголовок уточнен)
        fig, ax = plt.subplots(figsize=(10, 6))

        target = results[seq_name]["Tanh"][True][0].flatten()
        p_tanh = results[seq_name]["Tanh"][True][1].flatten()
        p_elu = results[seq_name]["ELU"][True][1].flatten()

        min_len = min(len(target), len(p_tanh), len(p_elu))
        err_tanh = np.abs(target[:min_len] - p_tanh[:min_len])
        err_elu = np.abs(target[:min_len] - p_elu[:min_len])

        ax.plot(err_tanh, 'r-', lw=1.5, alpha=0.8, label='Ошибка Tanh')
        ax.plot(err_elu, 'b-', lw=1.5, alpha=0.8, label='Ошибка ELU')
        # Заливка для наглядности
        ax.fill_between(range(min_len), err_tanh, color='red', alpha=0.1)
        ax.fill_between(range(min_len), err_elu, color='blue', alpha=0.1)

        ax.set_title(f"Сравнение абсолютной ошибки (при наличии нормализации): {seq_name}", fontsize=14,
                     fontweight='bold')
        ax.set_xlabel("Шаг последовательности (t)", fontsize=12)
        ax.set_ylabel("Ошибка |Target - Pred|", fontsize=12)
        ax.legend(frameon=True)

        save_plot(fig, f"{safe_name}_3_Error_Comparison.png")

        # --- ГРАФИК 3: СКОРОСТЬ ОБУЧЕНИЯ (LOSS) ---
        # (Четкое указание последовательности в заголовке)
        fig, ax = plt.subplots(figsize=(10, 6))

        loss_tanh = results[seq_name]["Tanh"][True][2]
        loss_elu = results[seq_name]["ELU"][True][2]

        ax.plot(loss_tanh, 'r--', label='Tanh Loss')
        ax.plot(loss_elu, 'b-', label='ELU Loss')
        ax.set_yscale('log')

        ax.set_title(f"Динамика обучения (Loss): {seq_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Эпохи обучения", fontsize=12)
        ax.set_ylabel("MSE Loss (Логарифмическая шкала)", fontsize=12)
        ax.legend(frameon=True)
        ax.grid(True, which="both", ls="-", alpha=0.2)

        save_plot(fig, f"{safe_name}_4_Learning_Curve.png")


if __name__ == "__main__":
    generate_images()