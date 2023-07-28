from typing import Callable

def adaptiveTrapezoidal(f: Callable, a: float, b: float, eps: float) -> float:
    """
    Computes definitive integral of function `f` on the interval `[a,b]`
    with adaptive trapezoidal rule with error rate at most `eps`.
    """
    h = b - a

    T1 = h * (f(a) + f(b)) / 2
    T2 = T1 / 2 + h / 2 * f((a + b) / 2)

    e = abs(T2 - T1) / 3
    if e < eps:
        return T2 + (T2 - T1) / 3
    else:
        T1 = adaptiveTrapezoidal(f, a, (a + b) / 2, eps / 2)
        T2 = adaptiveTrapezoidal(f, (a + b) / 2, b, eps / 2)
        return T1 + T2