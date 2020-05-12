import numpy as np
import matplotlib.pyplot as plt

def closed_loop(t, x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    K = 31
    I = 0.031
    Jm = 0.004
    ni1 = -K/I
    ni3 = K/I 
    ni4 = K/Jm
    ni5 = 1/Jm
    lambda1 = 10
    lambda2 = 10
    Kd1 = 100
    Kd2 = 100
    gamma1=10
    xd = 1 + np.sin(2*t+np.pi/2)*np.cos(3*t)
    xdponto = -2*np.sin(2*t)*np.cos(3*t)-3*np.sin(3*t)*np.cos(2*t);
    xd2ponto = 12*np.sin(2*t)*np.sin(3*t)-13*np.cos(2*t)*np.cos(3*t);
    xd3ponto = 62*np.sin(2*t)*np.cos(3*t)+63*np.cos(2*t)*np.sin(3*t);
    xd4ponto = -312*np.sin(2*t)*np.sin(3*t)+313*np.cos(2*t)*np.cos(3*t);
    f1 = ni1*x1
    f1ponto = ni1*x2
    f2 = ni4*(x1-x3)
    e1 = x1 - xd
    e1ponto = x2 - xdponto
    e2ponto = f1+ni3*x3 - xd2ponto
    e3ponto = f1ponto+ni3*x4 - xd3ponto
    xr1ponto = xdponto - lambda1*e1
    xr12ponto = xd2ponto - lambda1*e1ponto
    xr13ponto = xd3ponto - lambda1*e2ponto
    xr14ponto = xd4ponto - lambda1*e3ponto
    z1 = x2 - xr1ponto
    z1ponto = f1+ni3*x3 - xr12ponto
    z12ponto = f1ponto+ni3*x4 - xr13ponto
    ued = x5*xr12ponto - Kd1*z1
    x5ponto = -gamma1*xr12ponto*z1
    x52ponto = -gamma1*xr13ponto*z1 - gamma1*xr12ponto*z1ponto
    uedponto = x5*xr13ponto + x5ponto*xr12ponto - Kd1*z1ponto
    ued2ponto = x5*xr14ponto + 2*x5ponto*xr13ponto + x52ponto*xr12ponto - Kd1*z12ponto
    x3d = (1/K)*ued + x1
    x3dponto = (1/K)*uedponto + x2
    x3d2ponto = (1/K)*ued2ponto + f1+ni3*x3
    e3 = x3 - x3d
    e3ponto = x4 - x3dponto
    z3  = e3ponto + lambda2*e3
    v0 = x3d2ponto - lambda2*e3ponto
    u = Jm*v0 + K*(x3-x1) - Kd2*z3
    dx1dt=x2; 
    dx2dt=f1+ni3*x3
    dx3dt=x4;
    dx4dt=f2+ni5*u;
    dx5dt = -gamma1*xr12ponto*z1;
    return [dx1dt,dx2dt,dx3dt,dx4dt, dx5dt]

def solve_equation(x0, t, delta_t):
    x = x0
    x1, x2, x3, x4, x5 = [], [], [], [], []
    for i in range(t.size):
        dxdt = closed_loop(t[i], x)
        x = x + delta_t*np.array([dxdt[0], dxdt[1], dxdt[2], dxdt[3], dxdt[4]])
        x1.append(x[0])
        x2.append(x[1])
        x3.append(x[2])
        x4.append(x[3])
        x5.append(x[4])
    return x1, x2, x3, x4, x5

if __name__=='__main__':
    x0 = np.array([0,0,0,0,0])
    t0 = 0
    tf = 2
    delta_t = 1/500
    t = np.arange(t0, tf, delta_t)
    x1, x2, x3, x4, x5 = solve_equation(x0, t, delta_t)
    plt.plot(t, x1)
    plt.show()
