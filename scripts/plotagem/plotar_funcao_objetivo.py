import numpy as np
import plotly.graph_objects as go
from scripts.funcoes_para_minizar import w18

# Constantes de domínio
LIMITE_MIN = -500.0
LIMITE_MAX = 500.0
INTERVALO = 201 # Utilizando um passo 5 (de -500 a 500)

def plotar_funcao_objetivo():
    # Criando um grid de pontos (X, Y)
    x = np.linspace(LIMITE_MIN, LIMITE_MAX, INTERVALO)
    y = np.linspace(LIMITE_MIN, LIMITE_MAX, INTERVALO)
    X, Y = np.meshgrid(x, y)

    # Calculando os valores da função objetivo para cada ponto no grid
    Z = w18(X, Y)
    print("Minimo da função:", np.min(Z))

    # Plotando a superfície 3D com Plotly
    fig = go.Figure(data=[
        go.Surface(
            z=Z, 
            x=X[0], # Plotly infere automaticamente o eixo X
            y=Y[:, 0], # Plotly infere automaticamente o eixo Y
            colorscale='Viridis' # Escolha de uma escala de cores
        )
    ])

    fig.update_layout(
        title='Função Objetivo W18 (-W15 + W4)',
        scene=dict(
            xaxis_title='X axis',
            yaxis_title='Y axis',
            zaxis_title='Z axis (W18 value)'
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    fig.show()
