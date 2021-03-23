mport numpy as np
import scipy.integrate as sci
import pandas as pd
import matplotlib.pyplot as plt


eo = 2.0
so = 4.0
po = 0.0
km = 5.0
kf = 2.0
kr = 4.0
kcat = 6.0

def model(t, y):
    inp = y
    dy = np.zeros((3))
    dy[0] = -kf * inp[0] * inp[1] + (kr + kcat) * (eo - inp[0])
    dy[1] = -kf * inp[0] * inp[1] + kr * (eo - inp[0])
    dy[2] = kcat * (eo - inp[0])
    return dy

t_start, t_end = 0.0, 12.0

y = np.array([2.0,4.0,0.0])
Yres = sci.solve_ivp(model, [t_start, t_end], y, method = 'RK45', max_step = 0.01)

yy = pd.DataFrame(Yres.y).T
tt = np.linspace(t_start,t_end,yy.shape[0])
with plt.style.context('fivethirtyeight'): 
    plt.figure(1, figsize=(20,5))
    plt.plot(tt,yy,lw=8, alpha=0.5);
    plt.grid(axis='y')
    for j in range(3):
        plt.fill_between(tt,yy[j],0, alpha=0.2, label='y['+str(j)+']')
    plt.legend(prop={'size':20})
