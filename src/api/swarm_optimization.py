import matplotlib
import numpy as np
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('axes', titlesize=15)
matplotlib.rc('axes', labelsize=15)
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
from pyswarms.single.global_best import GlobalBestPSO
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

def closed_loop(t, x, robot, gains, adapt_gains):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]
    x8 = x[7]
    ni1 = -robot.K/robot.I
    ni2 =  robot.Mgl/robot.I
    ni3 = robot.K/robot.I 
    ni4 = robot.K/robot.Jm
    ni5 = 1.0/robot.Jm
    xd = 0.1*(1 + np.sin(2*t+np.pi/2)*np.cos(3*t))
    xdponto = 0.1*(-2*np.sin(2*t)*np.cos(3*t)-3*np.sin(3*t)*np.cos(2*t))
    xd2ponto = 0.1*(12*np.sin(2*t)*np.sin(3*t)-13*np.cos(2*t)*np.cos(3*t))
    xd3ponto = 0.1*(62*np.sin(2*t)*np.cos(3*t)+63*np.cos(2*t)*np.sin(3*t))
    xd4ponto = 0.1*(-312*np.sin(2*t)*np.sin(3*t)+313*np.cos(2*t)*np.cos(3*t))
    f1 = ni1*x1+ni2*np.sin(x1)
    f1ponto = ni1*x2 + ni2*np.cos(x1)*x2
    e1 = x1 - xd
    e1ponto = x2 - xdponto
    e2ponto = ni3*(x3-x1) + ni2*np.sin(x1) - xd2ponto
    e3ponto = ni3*(x4-x2) + ni2*np.cos(x1)*x2 - xd3ponto
    xr1ponto = xdponto - gains['lambda1']*e1
    xr12ponto = xd2ponto - gains['lambda1']*e1ponto
    xr13ponto = xd3ponto - gains['lambda1']*e2ponto
    xr14ponto = xd4ponto - gains['lambda1']*e3ponto
    z1 = x2 - xr1ponto
    z1ponto = ni3*(x3-x1)+ni2*np.sin(x1) - xr12ponto
    z12ponto = ni3*(x4-x2)+ni2*np.cos(x1)*x2 - xr13ponto
    x5ponto = -adapt_gains['gamma1']*xr12ponto*z1
    x52ponto = -adapt_gains['gamma1']*xr13ponto*z1 - adapt_gains['gamma1']*xr12ponto*z1ponto
    term1 = x5*xr12ponto
    term1ponto = x5ponto*xr12ponto + x5*xr13ponto
    term12ponto = x52ponto*xr12ponto + 2*x5ponto*xr13ponto + x5*xr14ponto
    x6ponto = adapt_gains['gamma2']*np.sin(x1)*z1
    x62ponto = adapt_gains['gamma2']*np.cos(x1)*x2*z1 + adapt_gains['gamma2']*np.sin(x1)*z1ponto
    term2 = x6*np.sin(x1)
    term2ponto = x6ponto*np.sin(x1) + x6*np.cos(x1)*x2
    term22ponto = x62ponto*np.sin(x1) + 2*x6ponto*np.cos(x1)*x2 + x6*(-np.sin(x1)*x2*x2 + np.cos(x1)*(f1+ni3*x3))
    ued = term1 - gains['kd1']*z1 - term2
    uedponto = term1ponto - gains['kd1']*z1ponto - term2ponto
    ued2ponto = term12ponto - gains['kd1']*z12ponto -term22ponto
    x3d = ued + x1
    x3dponto = uedponto + x2
    x3d2ponto = ued2ponto + f1+ni3*x3
    e3 = x3 - x3d
    e3ponto = x4 - x3dponto
    z3  = e3ponto  + gains['lambda2']*e3
    v0 = x3d2ponto - gains['lambda2']*e3ponto
    u = x7*v0 - x8*(x1-x3) - gains['kd2']*z3
    dx1dt = x2; 
    dx2dt = f1+ni3*x3
    dx3dt = x4
    dx4dt = ni4*(x1-x3)+ni5*u
    dx5dt = -adapt_gains['gamma1']*xr12ponto*z1
    dx6dt = adapt_gains['gamma2']*np.sin(x1)*z1
    dx7dt = -adapt_gains['gamma3']*v0*z3
    dx8dt = adapt_gains['gamma4']*(x1-x3)*z3
    return np.array([dx1dt,dx2dt,dx3dt,dx4dt,dx5dt,dx6dt,dx7dt,dx8dt])

def solve_equation(x, t, delta_t, robot, gains, adapt_gains):
    x_plot = []
    for i in range(t.size):
        k1 = closed_loop(t[i], x, robot, gains, adapt_gains)
        k2 = closed_loop(t[i] + 0.5*delta_t, x + 0.5*delta_t*k1, robot, gains, adapt_gains)
        k3 = closed_loop(t[i] + 0.5*delta_t, x + 0.5*delta_t*k2, robot, gains, adapt_gains)
        k4 = closed_loop(t[i] + delta_t, x + delta_t*k3, robot, gains, adapt_gains)
        x = x + (delta_t/6)*(k1+2*k2+2*k3+k4)
        x_plot.append(x)
    return np.array(x_plot)

def itae(gains, x0, xd, t):
    control_gains = {'lambda1': gains[0,0], 'lambda2': gains[1,0], 'kd1': gains[2,0],'kd2': gains[3,0]}
    adapt_gains = {'gamma1': gains[4,0], 'gamma2': gains[5,0], 'gamma3': gains[6,0],'gamma4': gains[7,0]}
    robot = Robot(K = 31, I = 31, Mgl = 0.8, Jm = 0.04)
    x1 = solve_equation(x0, t, delta_t, robot, control_gains, adapt_gains)[:,0]
    if np.isnan(x1[-1]):
        return float(randint(10**5,10**6))
    else:
        return sum(t*abs(xd-x1))

if __name__=='__main__':    
    Khat = 35
    Ihat = 30
    Jmhat = 0.03
    Mglhat = 1.0
    t0 = 0
    tf = 10
    delta_t = 1/1000
    t = np.arange(t0, tf, delta_t)
    x0 = np.array([0, 0, 0, 0, Ihat/Khat, Mglhat/Khat, Jmhat, Khat])
    xd = 0.1*(1 + np.sin(2*t+np.pi/2)*np.cos(3*t))
    gain_min = [0.1, 0.1, 0.1, 0.1, 0.02, 0.2, 0.001, 1000]
    gain_max = [10, 10, 10, 100, 0.1, 9, 0.2, 200000]
    bounds = (gain_min, gain_max)
    options = {'c1': 0.5, 'c2': 0.8, 'w': 0.9}
    optimizer = GlobalBestPSO(n_particles=10, dimensions=8, options=options,bounds=bounds)
    kwargs={"x0": x0, "xd": xd, 't': t}
    cost, pos = optimizer.optimize(itae, 1000, **kwargs)
    print(cost)
    print(pos)