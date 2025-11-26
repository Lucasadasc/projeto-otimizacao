import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scripts.funcoes_para_minizar import w18

# Constantes de domínio
LIMITE_MIN = -500.0
LIMITE_MAX = 500.0

def plotar_funcao_objetivo(intervalo=201):

    # 1. Criar um grid de pontos (X, Y)
    x = np.linspace(LIMITE_MIN, LIMITE_MAX, intervalo)
    y = np.linspace(LIMITE_MIN, LIMITE_MAX, intervalo)
    X, Y = np.meshgrid(x, y)

    # 2. Calcular os valores da função objetivo para cada ponto no grid
    Z = w18(X, Y)

    # 3. Plotar a superfície 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.viridis, linewidth=0, antialiased=False)

    ax.set_title('Função Objetivo W18')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis (W18 value)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()

if __name__ == "__main__":
    plotar_funcao_objetivo()
