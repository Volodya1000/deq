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
                #Вычисляет норму тензора. По умолчанию это L2-норма (евклидова)
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
        # Обратный проход и Неявное дифференцирование
        z_star, x = ctx.saved_tensors
        f = ctx.f

        z_star.requires_grad_()

        # Эта функция вычисляет векторно-якобиевское произведение в обратном направлении
        def g_rev(v):
            # v — текущая оценка вектора, который мы ищем
            with torch.enable_grad():       # включаем запись графа только внутри
                f_val = f(z_star, x)         # прямой проход один раз
            # Считаем ∇_z (f(z*, x)) · v   — это и есть векторно-якобиевское произведение
            vjp = torch.autograd.grad(
                outputs=f_val,
                inputs=z_star,
                grad_outputs=v,             # вот сюда подставляем вектор v
                retain_graph=True           # граф нужен ещё много раз
            )[0]
            # По формуле неявного дифференцирования: v* = grad_output + ∇_z f · v*
            return grad_output + vjp

        # Теперь решаем уравнение v* = grad_output + ∇_z f(z*, x) · v*  итеративно
        # (точно так же, как в прямом проходе искали z*)
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
