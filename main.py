# Лабораторная практическая работа 3 по дисциплине МРЗВИС
# Выполнена студентами группы 221701 БГУИР Дичковским Владимиром Андреевичем
# 08.12.2025
# Использованные материалы:
# Deep Equilibrium Models S. Bai, J.Z. Kolter, and V. Koltun.
# Advances in Neural Information Processing Systems (NeurIPS) 2019


import matplotlib.pyplot as plt
from data import get_sequences
from train import train_and_evaluate

def main():
    sequences = get_sequences()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (name, seq) in enumerate(sequences.items()):
        w_size = 3 if len(seq) < 30 else 10
        h_dim = 40
        n_epochs = 300 if len(seq) < 30 else 100

        result = train_and_evaluate(name, seq, window_size=w_size,
                                    hidden_dim=h_dim, epochs=n_epochs)

        if result:
            target, pred, _ = result
            ax = axes[i]
            ax.plot(target, 'b.-', label='Target', linewidth=1.5)
            ax.plot(pred, 'r.--', label='DEQ Prediction', linewidth=1.5)
            ax.set_title(f"{name} (Win={w_size})")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()