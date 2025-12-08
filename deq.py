# Лабораторная практическая работа 3 по дисциплине МРЗВИС
# Выполнена студентами группы 221701 БГУИР Дичковским Владимиром Андреевичем
# 08.12.2025
# Использованные материалы:
# Deep Equilibrium Models S. Bai, J.Z. Kolter, and V. Koltun.
# Advances in Neural Information Processing Systems (NeurIPS) 2019


import torch

class DEQFixedPoint(torch.autograd.Function):
    """Автодифференцируемая функция для поиска неподвижной точки"""

    @staticmethod
    def forward(ctx, f, x, z_init, threshold=1e-3, max_iter=50):
        # Прямой проход: ищем неподвижную точку z* = f(z*, x)
        with torch.no_grad(): #чтоб не сохранять промежуточных состояний
            z = z_init
            for i in range(max_iter):
                z_prev = z
                z = f(z, x)
                #Вычисляет норму тензора. По умолчанию это L2-норма
                if torch.norm(z - z_prev) < threshold:
                    break

        # Сохраняем для backward
        z_star = z
        ctx.save_for_backward(z_star, x)
        ctx.f = f
        ctx.max_iter = max_iter
        ctx.threshold = threshold
        return z_star

    @staticmethod
    def backward(ctx, grad_output):
        # Обратный проход: Implicit Differentiation
        z_star, x = ctx.saved_tensors
        f = ctx.f

        z_star.requires_grad_()

        # Функция для вычисления VJP
        def g_rev(v):
            with torch.enable_grad():
                f_val = f(z_star, x)
            vjp = torch.autograd.grad(f_val, z_star, grad_outputs=v, retain_graph=True)[0]
            return grad_output + vjp

        # Ищем решение обратной системы итеративно
        v = torch.zeros_like(grad_output)
        with torch.no_grad():
            for i in range(ctx.max_iter):
                v_prev = v
                v = g_rev(v)
                if torch.norm(v - v_prev) < ctx.threshold:
                    break

        # Финальный градиент по параметрам и входу
        with torch.enable_grad():
            f_final = f(z_star, x)
        torch.autograd.backward(f_final, v)

        return None, x.grad, None, None, None