
import numpy as np
import matplotlib.pyplot as plt

def f(t, x):
    return x+1


def main(t, x0, delta_t):
    x = []
    x.append(x0)
    x_n = x0
    for i in range(t.shape[0]-1):

        k1 = f(t[i], x_n)
        k2 = f(t[i] + 0.5*delta_t, x_n + 0.5*delta_t*k1)
        k3 = f(t[i] + 0.5*delta_t, x_n + 0.5*delta_t*k2)
        k4 = f(t[i] + delta_t, x_n + delta_t*k3)
        x_n = x_n + (delta_t/6)*(k1+2*k2+2*k3+k4)
        x.append(x_n)
    return x
if __name__=='__main__':
    x0 = 0
    delta_t = 0.02
    tf = 10
    t = np.arange(0, tf, delta_t)
    x = main(t, x0, delta_t)
    print(x)
    print(t)
    plt.plot(t, x)
    plt.plot(t, np.exp(t)-1)
    plt.show()
