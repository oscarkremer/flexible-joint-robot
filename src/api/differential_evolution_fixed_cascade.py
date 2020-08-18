import matplotlib
import numpy as np
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('axes', titlesize=15)
matplotlib.rc('axes', labelsize=15)
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
from scipy.optimize import differential_evolution
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
from random import randint

class Robot:
    def __init__(self, K, I, Mgl, Jm):
        self.K = K
        self.I = I 
        self.Mgl = Mgl 
        self.Jm = Jm

def closed_loop(t, x, robot, robot_hat, gains):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    ni1 = -robot.K/robot.I
    ni2 =  robot.Mgl/robot.I
    ni3 = robot.K/robot.I 
    ni4 = robot.K/robot.Jm
    ni5 = 1.0/robot.Jm
    lambda1 = gains['lambda1']
    lambda2 = gains['lambda2']
    Kd1 = gains['kd1']
    Kd2 = gains['kd2']
    xd = 0.1*(1 + np.sin(2*t+np.pi/2)*np.cos(3*t))
    xdponto = 0.1*(-2*np.sin(2*t)*np.cos(3*t)-3*np.sin(3*t)*np.cos(2*t))
    xd2ponto = 0.1*(12*np.sin(2*t)*np.sin(3*t)-13*np.cos(2*t)*np.cos(3*t))
    xd3ponto = 0.1*(62*np.sin(2*t)*np.cos(3*t)+63*np.cos(2*t)*np.sin(3*t))
    xd4ponto = 0.1*(-312*np.sin(2*t)*np.sin(3*t)+313*np.cos(2*t)*np.cos(3*t))
    f1 = ni1*x1 + ni2*np.sin(x1)
    f1ponto = ni1*x2 + ni2*np.cos(x1)*x2
    e1 = x1 - xd
    e1ponto = x2 - xdponto
    e2ponto = ni3*(x3-x1) + ni2*np.sin(x1) - xd2ponto
    e3ponto = ni3*(x4-x2) + ni2*np.cos(x1)*x2 - xd3ponto
    xr1ponto = xdponto - lambda1*e1
    xr12ponto = xd2ponto - lambda1*e1ponto
    xr13ponto = xd3ponto - lambda1*e2ponto
    xr14ponto = xd4ponto - lambda1*e3ponto
    z1 = x2 - xr1ponto
    z1ponto = ni3*(x3-x1) + ni2*np.sin(x1) - xr12ponto
    z12ponto = ni3*(x4-x2) + ni2*np.cos(x1)*x2 - xr13ponto
    ued = robot_hat.I*xr12ponto - Kd1*robot_hat.K*z1
    uedponto = robot_hat.I*xr13ponto - Kd1*robot_hat.K*z1ponto
    ued2ponto = robot_hat.I*xr14ponto - Kd1*robot_hat.K*z12ponto
    x3d = (1/robot_hat.K)*ued + x1 - robot_hat.Mgl*np.sin(x1)
    x3dponto = (1/robot_hat.K)*uedponto + x2 - robot_hat.Mgl*np.cos(x1)*x2
    x3d2ponto = (1/robot_hat.K)*ued2ponto + f1+ni3*x3 + robot_hat.Mgl*np.sin(x1)*(x2**2) - robot_hat.Mgl*np.cos(x1)*(ni3*(x3-x1) + ni2*np.sin(x1))
    e3 = x3 - x3d
    e3ponto = x4 - x3dponto
    z3  = e3ponto + lambda2*e3
    v0 = x3d2ponto - lambda2*e3ponto
    u = robot_hat.Jm*v0 - robot_hat.K*(x1-x3) - Kd2*z3
    dx1dt=x2; 
    dx2dt=ni3*(x3-x1) + ni2*np.sin(x1)
    dx3dt=x4
    dx4dt=ni4*(x1-x3) + ni5*u
    return np.array([dx1dt,dx2dt,dx3dt,dx4dt])

def solve_equation(x, t, delta_t, robot, robot_hat, gains):
    x_plot = []
    for i in range(t.size):
        k1 = closed_loop(t[i], x, robot, robot_hat, gains)
        k2 = closed_loop(t[i] + 0.5*delta_t, x + 0.5*delta_t*k1, robot, robot_hat, gains)
        k3 = closed_loop(t[i] + 0.5*delta_t, x + 0.5*delta_t*k2, robot, robot_hat, gains)
        k4 = closed_loop(t[i] + delta_t, x + delta_t*k3, robot, robot_hat, gains)
        x = x + (delta_t/6)*(k1+2*k2+2*k3+k4)
        x_plot.append(x)
    return np.array(x_plot)

def itae(gains):
    Khat = 45
    Ihat = 45
    Jmhat = 0.06
    Mglhat = 1.2
    t0 = 0
    tf = 10
    delta_t = 1/1000
    t = np.arange(t0, tf, delta_t)
    x0 = np.array([0, 0, 0, 0])
    xd = 0.1*(1 + np.sin(2*t+np.pi/2)*np.cos(3*t))
    control_gains = {'lambda1': gains[0], 'lambda2': gains[1], 'kd1': gains[2],'kd2': gains[3]}
    robot_hat = Robot(K = Khat, I = Ihat, Mgl = Mglhat, Jm = Jmhat)

    robot = Robot(K = 31, I = 31, Mgl = 0.8, Jm = 0.04)
    x1 = solve_equation(x0, t, delta_t, robot, robot_hat, control_gains)[:,0]
    if np.isnan(x1[-1]):
        return float(randint(10**5,10**6))
    else:
        metric = sum(t*abs(xd-x1))
        return metric

if __name__=='__main__':    
    bounds = [(0.1, 20), (0.1, 20), (0.1, 20), (0.1, 200)]
    result = differential_evolution(itae, bounds, maxiter=100, popsize=10, disp=True)
    print(result.x)
    print(result.fun)

#[20.         20.         20.         24.15402461] - com erro - itae 778.9458416330021
#[ 20.          20.          20.         108.69215305] - sem erro - itae -14.745204873209527

