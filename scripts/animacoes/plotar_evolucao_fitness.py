import os
import plotly.graph_objects as go

def plotar_evolucao_fitness(dados_algoritmo, algoritmo_selecionado="pso", salvar_grafico=True):
    """
    Plota a evolução do fitness ao longo das iterações/gerações.
    
    Parâmetros:
    - algoritmo_selecionado: "pso" ou "ag"
    - salvar_grafico: se deve salvar como arquivo HTML
    """
    
    if algoritmo_selecionado == "pso":
        print("Executando PSO para plotar evolução do fitness...")
        historico = dados_algoritmo["historico"]
        melhor_valor = dados_algoritmo["melhor_valor"]
        iteracoes = dados_algoritmo["quantidade_iteracoes_realizadas"]
        titulo_principal = "Evolução do Fitness - PSO"
        eixo_x_titulo = "Iterações"
        nome_arquivo = "evolucao_fitness_pso.html"
    else:
        print("Executando Algoritmo Genético para plotar evolução do fitness...")
        historico = dados_algoritmo["historico_fitness"]
        melhor_valor = dados_algoritmo["melhor_valor"]
        iteracoes = dados_algoritmo["quantidade_geracoes_realizadas"]
        titulo_principal = "Evolução do Fitness - Algoritmo Genético"
        eixo_x_titulo = "Gerações"
        nome_arquivo = "evolucao_fitness_ag.html"
    
    print(f"Criando gráfico de evolução do fitness ({algoritmo_selecionado.upper()})...")
    
    # Criar o gráfico de linha
    fig = go.Figure()
    
    # Linha principal da evolução
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(historico) + 1)),
            y=historico,
            mode='lines+markers',
            name='Fitness por Iteração',
            line=dict(color='blue', width=2),
            marker=dict(size=6, color='blue'),
            hovertemplate=f'{eixo_x_titulo}: %{{x}}<br>Fitness: %{{y:.2f}}<extra></extra>'
        )
    )
    
    # Linha do melhor valor final (referência horizontal)
    fig.add_trace(
        go.Scatter(
            x=[1, len(historico)],
            y=[melhor_valor, melhor_valor],
            mode='lines',
            name=f'Melhor Valor Final: {melhor_valor:.2f}',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate=f'Melhor Valor: {melhor_valor:.2f}<extra></extra>'
        )
    )
    
    # Destacar o ponto do melhor valor
    melhor_indice = historico.index(min(historico)) + 1
    fig.add_trace(
        go.Scatter(
            x=[melhor_indice],
            y=[melhor_valor],
            mode='markers',
            name=f'Melhor em {eixo_x_titulo} {melhor_indice}',
            marker=dict(
                size=12,
                color='gold',
                symbol='star',
                line=dict(color='red', width=2)
            ),
            hovertemplate=f'{eixo_x_titulo}: {melhor_indice}<br>Melhor Fitness: {melhor_valor:.2f}<extra></extra>'
        )
    )
    
    # Configurar layout
    fig.update_layout(
        title=dict(
            text=f'{titulo_principal}<br>'
                 f'Melhor valor: {melhor_valor:.2f} ({eixo_x_titulo.lower()} realizadas: {iteracoes})',
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title=eixo_x_titulo,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Valor da Função Objetivo (W18)',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        template='plotly_white',
        height=600,
        width=900,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Adicionar anotações estatísticas
    melhoria_total = historico[0] - melhor_valor
    melhoria_percentual = (melhoria_total / abs(historico[0])) * 100 if historico[0] != 0 else 0
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.85,
        text=f"<b>Estatísticas:</b><br>"
             f"Valor inicial: {historico[0]:.2f}<br>"
             f"Valor final: {melhor_valor:.2f}<br>"
             f"Melhoria absoluta: {melhoria_total:.2f}<br>"
             f"Melhoria percentual: {melhoria_percentual:.2f}%",
        showarrow=False,
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    if salvar_grafico:
        if not nome_arquivo.endswith('.html'):
            nome_arquivo += '.html'
        
        caminho_completo = os.path.join(os.getcwd(), 'assets/graficos/', nome_arquivo)
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(caminho_completo), exist_ok=True)
        
        fig.write_html(caminho_completo)
        print(f"Gráfico de evolução salvo como: {caminho_completo}")
        
        print("Deseja abrir o arquivo agora? (s/n)")
        resposta = input().strip().lower()
        if resposta == 's':
            os.startfile(caminho_completo)
    else:
        fig.show()
    
    return fig
