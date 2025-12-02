import numpy as np
import pandas as pd
from scripts.algoritmos.pso.pso import calcular_pso

def testar_ag_metricas_completas():
    """
    Testa o PSO variando parâmetros individualmente e coletando métricas específicas:
    - Média de gBest
    - Desvio Padrão do gBest
    - Média de Avaliações da Função
    - Taxa de Sucesso (≥75% convergência para melhor gBest)
    
    Cada parâmetro é variado individualmente com 10 execuções por teste.
    """
    
    numero_individuos = [20, 30, 40, 60, 80]
    taxa_crossover = [0.6, 0.7, 0.8, 0.9, 1.0]
    taxa_mutacao = 0.1
    intensidade_mutacao = 10
    num_execucoes_por_teste = 15

    resultados = []

    # VARIANDO O NÚMERO DE PARTÍCULAS
    for num_particulas in numero_individuos:
        for i in range(5):
            gbest_values = []
            avaliacoes_funcao = []
            sucessos = 0

            taxa_crossover_atual = taxa_crossover[i] 

            for _ in range(num_execucoes_por_teste):
                from scripts.algoritmos.genetico.calcular_algoritmo_genetico import calcular_algoritmo_genetico
                dados_ag = calcular_algoritmo_genetico(
                    numero_individuos=num_particulas,
                    taxa_crossover=taxa_crossover_atual,
                    taxa_mutacao=taxa_mutacao,
                    intensidade_mutacao=intensidade_mutacao,
                    limites=(-500, 500),
                )
                gbest_values.append(dados_ag['melhor_valor'])
                avaliacoes_funcao.append(dados_ag['numero_execucoes_funcao_objetivo'])
                sucesso = dados_ag.get('funcao_convergiu_para_minimo_esperado', False)
                if sucesso:
                    sucessos += 1
                
            media_gbest = np.mean(gbest_values)
            desvio_padrao_gbest = np.std(gbest_values)
            media_avaliacoes = np.mean(avaliacoes_funcao)
            taxa_sucesso = (sucessos / num_execucoes_por_teste)
            resultados.append({
                'parametro': 'num_individuos',
                'valor': num_particulas,
                'media_gbest': round(media_gbest, 2),
                'desvio_padrao_gbest': round(desvio_padrao_gbest, 2),
                'media_avaliacoes_funcao': round(media_avaliacoes, 2),
                'taxa_sucesso_percentual': round(taxa_sucesso * 100, 2),
                'valor_taxa_crossover': taxa_crossover_atual,
                'intensidade_mutacao': intensidade_mutacao,
                'taxa_mutacao': taxa_mutacao
            })

    # printando resultados em formato tabular:
    df_resultados = pd.DataFrame(resultados)
    print(df_resultados)
    # printando resultados em formato json 
    print(df_resultados.to_json(orient='records', indent=4))
    
testar_ag_metricas_completas()