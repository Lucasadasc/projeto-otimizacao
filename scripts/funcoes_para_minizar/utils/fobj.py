
import numpy as np

a=500
b=0.1
c=0.5*np.pi # c=0.5*%pi;

# Fobj=F10.*zsh//+a*cos(x1/30);
def fobj(x, y):
    f10 = calcular_f10(x, y)
    zsh = calcular_zhs()

    fobj = f10 * zsh 
        
    return fobj

def calcular_f10(x, y):
    x_normalizado = x / 250
    y_normalizado = y / 250

    x1=25*x_normalizado
    x2=25*y_normalizado
    
    # F10=-a*exp(-b*sqrt((x1.^2+x2.^2)/2))-exp((cos(c*x1)+cos(c*x2))/2)+exp(1);
    f10 = -a * np.exp(-b * np.sqrt((x1**2 + x2**2) / 2)) - np.exp((np.cos(c * x1) + np.cos(c * x2)) / 2) + np.exp(1)

    return f10

# zsh(i,j)=0.5-((sin(sqrt(xs(i)^2+ys(j)^2)))^2-0.5)./(1+0.1*(xs(i)^2+ys(j)^2))^2;
def calcular_zhs():
    xs = np.asarray(np.arange(-10, 10.1, 0.1)) # xs =-10:0.1:10;
    ys = np.asarray(np.arange(-10, 10.1, 0.1)) # o as transforma em array numpy ys =-10:0.1:10;

    numerador = np.sin(np.sqrt(xs**2 + ys**2))**2 -0.5
    denominador = (1+0.1*(xs**2 + ys**2))**2

    return 0.5 - numerador / denominador

