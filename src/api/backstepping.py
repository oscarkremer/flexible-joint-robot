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
    ni1hat = -robot_hat.K/robot_hat.I
    ni2hat =  robot_hat.Mgl/robot_hat.I
    ni3hat = robot_hat.K/robot_hat.I
    ni4hat = robot_hat.K/robot_hat.Jm
    ni5hat = 1.0/robot_hat.Jm
    xd = 0.1*(1 + np.sin(2*t+np.pi/2)*np.cos(3*t))
    xdponto = 0.1*(-2*np.sin(2*t)*np.cos(3*t)-3*np.sin(3*t)*np.cos(2*t))
    xd2ponto = 0.1*(12*np.sin(2*t)*np.sin(3*t)-13*np.cos(2*t)*np.cos(3*t))
    xd3ponto = 0.1*(62*np.sin(2*t)*np.cos(3*t)+63*np.cos(2*t)*np.sin(3*t))
    xd4ponto = 0.1*(-312*np.sin(2*t)*np.sin(3*t)+313*np.cos(2*t)*np.cos(3*t))
    f1 = ni1*x1 + ni2*np.sin(x1)
    x1dot=x2
    x2dot=f1+ni3*x3
    x3dot=x4
    f1ponto = ni1*x2 + ni2*np.cos(x1)*x2
    f12ponto = ni1*x2dot - ni2*np.sin(x1)*x2**2 + ni2*np.cos(x1)*x2dot
    f2 = ni4*(x1-x3)
    e1 = x1 - xd
    e1ponto = x2 - xdponto
    e12ponto = ni3*(x3-x1) + ni2*np.sin(x1) - xd2ponto
    e13ponto = ni3*(x4-x2) + ni2*np.cos(x1)*x2 - xd3ponto
    phi1 = xdponto-gains[0]*e1
    phi1ponto = xd2ponto-gains[0]*e1ponto
    phi12ponto = xd3ponto-gains[0]*e12ponto
    phi13ponto = xd4ponto-gains[0]*e13ponto
    e2 = x2-phi1; 
    e2ponto = x2dot-phi1ponto; 
    e22ponto = f1ponto+ni3*x4-phi12ponto; 
    phi2 = (1/ni3hat)*(-f1 + phi1ponto - e1 - gains[1]*e2)
    phi2ponto = (1/ni3hat)*(-f1ponto + phi12ponto - e1ponto - gains[1]*e2ponto)
    phi22ponto = (1/ni3hat)*(-f12ponto + phi13ponto - e12ponto - gains[1]*e22ponto)
    e3 = x3-phi2; 
    e3ponto = x4-phi2ponto; 
    phi3 = phi2ponto-ni3hat*e2-gains[2]*e3
    phi3ponto = phi22ponto-ni3hat*e2ponto-gains[2]*e3ponto
    e4 = x4-phi3
    u = (1.0/ni5hat)*(-f2+phi3ponto-e3-gains[3]*e4)
    dx1dt = x2 
    dx2dt = ni3*(x3-x1) + ni2*np.sin(x1)
    dx3dt = x4
    dx4dt = ni4*(x1-x3) + ni5*u
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
    t0 = 0
    tf = 10
    delta_t = 1/1000
    t = np.arange(t0, tf, delta_t)
    x0 = np.array([0,0,0,0])
    robot = Robot(K = 31, I = 3.1, Mgl = 0.8, Jm = 0.04)
    robot_hat = Robot(K = 35, I = 3.0, Mgl = 1.0, Jm = 0.03)
    gains = [10, 10, 10, 10]
    x_plot = solve_equation(x0, t, delta_t, robot, robot_hat, gains)
    xd = 0.1*(1 + np.sin(2*t+np.pi/2)*np.cos(3*t))
    xdponto = 0.1*(-2*np.sin(2*t)*np.cos(3*t)-3*np.sin(3*t)*np.cos(2*t))
    print('ITAE - {}'.format(itae(x_plot[:,0], xd, t)))
    plot(t, x_plot[:,0], xd, 'Controlador por Backstepping com Incertezas Paramétricas', 'tempo(segundos)', 
        'Posicao Angular', ['Posição Angular', 'Referência'])
    plot(t, x_plot[:,0]-xd, x_plot[:,1]-xdponto, 'Controlador por Backstepping com Incertezas Paramétricas', 
        'tempo(segundos)', 'Erros', ['Erro de Posição', 'Erro de Velocidade'], axis_lim=True)



