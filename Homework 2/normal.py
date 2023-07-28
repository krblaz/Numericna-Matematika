import math
import sys
from adaptive_trapezoidal import *

def normal(x: float, eps: float = 1e-10) -> float:
    """
    Computes probability of normal distribution for `x` with error rate at most `eps`
    """
    f = lambda t: math.exp(-(t**2) / 2)

    res = 0.5
    if x > 0:
        res = 0.5 + adaptiveTrapezoidal(f, 0, x, eps) / math.sqrt(2 * math.pi)
    elif x < 0:
        res = 0.5 - adaptiveTrapezoidal(f, x, 0, eps) / math.sqrt(2 * math.pi)
    return res


if __name__ == "__main__":
    try:
        x = float(sys.argv[1])
        eps = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-10
        print(normal(x,eps))
    except:
        print(
            f"Usage: {sys.argv[0]} <p> [eps = 10e-10] where x is input of "
            "the normal distribution and eps is desired maximum error"
        )
