import matplotlib
import numpy as np
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('axes', titlesize=15)
matplotlib.rc('axes', labelsize=15)
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

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

def itae(x1, xd, t):
    return sum(t*abs(xd-x1))

def plot(t, y1, y2, title, x_label, y_label, legends, axis_lim=False):
    axis_font = {'fontname':'Arial', 'size':'15'}   
    plt.figure(figsize=(10,5))
    plt.plot(t, y1)
    plt.plot(t, y2)
    plt.title(title)
    plt.xlabel(x_label, **axis_font)
    plt.ylabel(y_label, **axis_font)
    plt.legend(legends, fontsize=15)
    if axis_lim:
        plt.xlim(min(t), max(t))
        plt.ylim(-1, 1)
    plt.show()

if __name__=='__main__':    
    Khat = 35
    Ihat = 30
    Jmhat = 0.03
    Mglhat = 1.0
    t0 = 0
    tf = 10
    delta_t = 1/1000
    t = np.arange(t0, tf, delta_t)
    robot = Robot(K = 31, I = 31, Mgl = 0.8, Jm = 0.04)
    gains = {'lambda1': 5, 'lambda2': 7.5, 'kd1': 5,'kd2': 75}
    adapt_gains = {'gamma1': 0.04, 'gamma2':4.5, 'gamma3': 0.1,'gamma4': 100000}
    x0 = np.array([0, 0, 0, 0, Ihat/Khat, Mglhat/Khat, Jmhat, Khat])
    x_plot = solve_equation(x0, t, delta_t, robot, gains, adapt_gains)
    xd = 0.1*(1 + np.sin(2*t+np.pi/2)*np.cos(3*t))
    xdponto = 0.1*(-2*np.sin(2*t)*np.cos(3*t)-3*np.sin(3*t)*np.cos(2*t))
    print('ITAE - {}'.format(itae(x_plot[:,0], xd, t)))
    plot(t, x_plot[:,0], xd, 'Controlador em Cascata Adaptativo', 'tempo(segundos)', 
        'Posicao Angular', ['Posição Angular', 'Referência'])
    plot(t, x_plot[:,0]-xd, x_plot[:,1]-xdponto, 'Controlador em Cascata Adaptativo', 
        'tempo(segundos)', 'Erros', ['Erro de Posição', 'Erro de Velocidade'], axis_lim=True)
    plot(t, np.ones(t.shape[0])*x0[4], x_plot[:,4], 'Controlador em Cascata Adaptativo', 
        'tempo(segundos)', 'Adaptação I/K', ['I/K', 'Adaptação'])
    plot(t, np.ones(t.shape[0])*x0[5], x_plot[:,5], 'Controlador em Cascata Adaptativo', 
    'tempo(segundos)', 'Adaptação Mgl/K', ['Mgl/K', 'Adaptação'])
    plot(t, np.ones(t.shape[0])*x0[6], x_plot[:,6], 'Controlador em Cascata Adaptativo', 
    'tempo(segundos)', 'Adaptação Jm', ['Jm', 'Adaptação'])
    plot(t, np.ones(t.shape[0])*x0[7], x_plot[:,7], 'Controlador em Cascata Adaptativo', 
    'tempo(segundos)', 'Adaptação K', ['K', 'Adaptação'])