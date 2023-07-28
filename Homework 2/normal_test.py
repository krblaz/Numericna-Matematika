from normal import *
import scipy.stats

if __name__ == "__main__":
    eps = 1e-10
    for x in range(-3,3,1):
        real = scipy.stats.norm().cdf(x)
        our = normal(x, eps)
        if abs(real - our) < eps:
            print(f"Test for input {x} PASSED")
        else:
            print(f"Test for input {x} FAILED")