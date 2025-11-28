import os
import numpy as np
import plotly.graph_objects as go
from scripts.funcoes_para_minizar import w18
from scripts.algoritmos import calcular_pso, calcular_algoritmo_genetico
from constants import constantes_pso, constantes_ag

def criar_animacao_particulas(
    algoritmo_selecionado="pso",
    duracao_frame=500,
    salvar_gif=True,
):
    """
    Cria uma animação da movimentação das partículas do PSO.
    
    Parâmetros:
    - num_particulas: número de partículas no enxame
    - max_iteracoes: número máximo de iterações
    - limites: tupla (min, max) definindo o espaço de busca
    - salvar_gif: se deve salvar como arquivo HTML
    - nome_arquivo: nome do arquivo a ser salvo
    - duracao_frame: duração de cada frame em milissegundos

    """
    
    limites = constantes_pso.LIMITES_ESPACO_BUSCA
    num_particulas = constantes_pso.NUMERO_PARTICULAS
    max_iteracoes = constantes_pso.MAX_ITERACOES
    nome_arquivo = f"animacao_posicoes_{algoritmo_selecionado}.html"

    if algoritmo_selecionado == "pso":
        print("Executando PSO para criar animação...")
        dados_algoritmo = calcular_pso(capturar_posicoes=True)

        melhor_posicao = dados_algoritmo["melhor_posicao"]
        melhor_valor = dados_algoritmo["melhor_valor"]
        historico = dados_algoritmo["historico"]
        posicoes_particulas = dados_algoritmo["posicoes_particulas"]
        quantidade_iteracoes_realizadas = dados_algoritmo["quantidade_iteracoes_realizadas"]

        dados_algoritmo
    else:
        print("Executando Algoritmo Genético para criar animação...")
        dados_algoritmo = calcular_algoritmo_genetico(salvar_posicoes=True)

        num_particulas = constantes_ag.NUMERO_INDIVIDUOS
        max_iteracoes = constantes_ag.NUMERO_GERACOES
        melhor_posicao = dados_algoritmo["melhor_posicao"]
        melhor_valor = dados_algoritmo["melhor_valor"]
        historico = dados_algoritmo["historico_fitness"]
        posicoes_particulas = dados_algoritmo["posicoes_populacao"]
        quantidade_iteracoes_realizadas = dados_algoritmo["quantidade_geracoes_realizadas"]

    print("Criando animação...")
    
    resolucao = 201 
    x = np.linspace(limites[0], limites[1], resolucao)
    y = np.linspace(limites[0], limites[1], resolucao)
    X, Y = np.meshgrid(x, y)
    Z = w18(X, Y)

    frames = []    
    cores = [
        'red', 'blue', 'lime', 'orange', 'purple', 'brown', 'pink', 'gray',
        'olive', 'cyan', 'magenta', 'yellow', 'lightblue', 'lightgreen',
        'lightcoral', 'gold', 'violet', 'turquoise', 'salmon', 'khaki'
    ]
    
    for iteracao in range(len(posicoes_particulas)):
        frame_data = []
        
        # Adicionar contorno de fundo
        frame_data.append(
            go.Contour(
                z=Z,
                x=x,
                y=y,
                colorscale='Viridis',
                opacity=0.6,
                showscale=False,
                contours=dict(showlabels=False),
                hovertemplate='x: %{x}<br>y: %{y}<br>W18: %{z}<extra></extra>'
            )
        )
        
        # Adicionar partículas da iteração atual
        for i in range(num_particulas):
            pos_x = posicoes_particulas[iteracao][i][0]
            pos_y = posicoes_particulas[iteracao][i][1]
            cor = cores[i % len(cores)]
            
            frame_data.append(
                go.Scatter(
                    x=[pos_x],
                    y=[pos_y],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=cor,
                        symbol='circle',
                        line=dict(color='black', width=1)
                    ),
                    name=f'P{i+1}' if iteracao == 0 else '',
                    showlegend=(iteracao == 0 and i < 8),
                    hovertemplate=f'Partícula {i+1}<br>x: %{{x}}<br>y: %{{y}}<extra></extra>'
                )
            )
        
        # Adicionar trajetórias até a iteração atual
        for i in range(num_particulas):
            if iteracao > 0:  # Só mostrar trajetória se houver pelo menos 2 pontos
                x_traj = [posicoes_particulas[iter][i][0] for iter in range(iteracao + 1)]
                y_traj = [posicoes_particulas[iter][i][1] for iter in range(iteracao + 1)]
                cor = cores[i % len(cores)]
                
                frame_data.append(
                    go.Scatter(
                        x=x_traj,
                        y=y_traj,
                        mode='lines',
                        line=dict(color=cor, width=2, dash='dot'),
                        opacity=0.6,
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
        
        # Adicionar melhor posição global se já foi encontrada
        if iteracao < len(historico):
            # Encontrar a melhor posição até esta iteração
            melhor_atual = min(historico[:iteracao+1])
            if melhor_atual < float('inf'):
                frame_data.append(
                    go.Scatter(
                        x=[melhor_posicao[0]],
                        y=[melhor_posicao[1]],
                        mode='markers',
                        marker=dict(
                            size=16,
                            color='gold',
                            symbol='star',
                            line=dict(color='black', width=2)
                        ),
                        name='Melhor Global' if iteracao == 0 else '',
                        showlegend=(iteracao == 0),
                        hovertemplate=f'Melhor Solução<br>x: %{{x}}<br>y: %{{y}}<br>Valor: {melhor_valor:.4f}<extra></extra>'
                    )
                )
        
        frames.append(go.Frame(
            data=frame_data,
            name=str(iteracao),
            layout=dict(
                title=f'PSO - Iteração {iteracao + 1}/{len(posicoes_particulas)}<br>'
                      f'Melhor valor: {historico[iteracao]:.6f}' if iteracao < len(historico) else ''
            )
        ))
    
    # Criar figura inicial
    fig = go.Figure(
        data=frames[0].data if frames else [],
        frames=frames
    )
    
    # Configurar layout
    fig.update_layout(
        title=f'Animação PSO - Função W18<br>'
              f'{num_particulas} partículas, {max_iteracoes} iterações<br>'
              f'Melhor valor final: {melhor_valor:.6f}',
        xaxis=dict(
            title='X',
            range=[limites[0], limites[1]],
            constrain='domain'
        ),
        yaxis=dict(
            title='Y',
            range=[limites[0], limites[1]],
            scaleanchor='x',
            scaleratio=1
        ),
        template='plotly_white',
        height=700,
        width=800,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"frame": {"duration": duracao_frame, "redraw": True},
                               "fromcurrent": True, "transition": {"duration": 50}}],
                        label="▶ Play",
                        method="animate"
                    ),
                    dict(
                        args=[{"frame": {"duration": 0, "redraw": True},
                               "mode": "immediate", "transition": {"duration": 0}}],
                        label="⏸ Pause",
                        method="animate"
                    )
                ]),
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.011,
                xanchor="right",
                y=0,
                yanchor="top"
            )
        ],
        sliders=[dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(
                font=dict(size=20),
                prefix="Iteração:",
                visible=True,
                xanchor="right"
            ),
            transition=dict(duration=50, easing="cubic-in-out"),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.1,
            y=0,
            steps=[dict(
                args=[[f.name], {"frame": {"duration": duracao_frame, "redraw": True},
                                 "mode": "immediate", "transition": {"duration": 50}}],
                label=str(i+1),
                method="animate"
            ) for i, f in enumerate(frames)]
        )]
    )
    
    if salvar_gif:
        if not nome_arquivo.endswith('.html'):
            nome_arquivo += '.html'
        
        caminho_completo = os.path.join(os.getcwd(), 'assets/animacoes/particulas/' + nome_arquivo)
        print(caminho_completo)
        fig.write_html(caminho_completo)
        print(f"Animação salva como: {caminho_completo}")
        
        print(f"O arquivo foi salvo. Foram necessarias {quantidade_iteracoes_realizadas} iterações para criar a animação.")
        
        print("Deseja abrir o arquivo agora? (s/n)")
        resposta = input().strip().lower()
        if resposta == 's':
            os.startfile(caminho_completo)
    else:
        fig.show()
    
    return melhor_posicao, melhor_valor, historico
