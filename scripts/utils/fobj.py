
import numpy as np


a=500
b=0.1
c=0.5*np.pi # c=0.5*%pi;

# Fobj=F10.*zsh//+a*cos(x1/30);
def fobj(x, y):
    f10 = calcular_f10(x, y)
    zsh = calcular_zhs()

    x1=25*x
    fobj = f10 * zsh + a * np.cos(x1 / 30)
    
    return fobj

def calcular_f10(x, y):
    x1=25*x
    x2=25*y
    
    # F10=-a*exp(-b*sqrt((x1.^2+x2.^2)/2))-exp((cos(c*x1)+cos(c*x2))/2)+exp(1);
    f10 = -a * np.exp(-b * np.sqrt((x1**2 + x2**2) / 2)) - np.exp((np.cos(c * x1) + np.cos(c * x2)) / 2) + np.exp(1)

    return f10

def calcular_zhs():
    xs = np.arange(-10, 10.1, 0.1) # xs =-10:0.1:10;
    ys = np.arange(-10, 10.1, 0.1) # ys =-10:0.1:10;

    nx = len(xs)
    ny = len(ys)

    zsh = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            zsh[i, j] = 0.5 - ((np.sin(np.sqrt(xs[i]**2 + ys[j]**2)))**2 - 0.5) / (1 + 0.1 * (xs[i]**2 + ys[j]**2))**2

    return zsh
