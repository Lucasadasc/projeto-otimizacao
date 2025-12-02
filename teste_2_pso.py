import numpy as np
import pandas as pd
from scripts.algoritmos.pso.pso import calcular_pso

def testar_pso_metricas_completas():
    """
    Testa o PSO variando parâmetros individualmente e coletando métricas específicas:
    - Média de gBest
    - Desvio Padrão do gBest
    - Média de Avaliações da Função
    - Taxa de Sucesso (≥75% convergência para melhor gBest)
    
    Cada parâmetro é variado individualmente com 10 execuções por teste.
    """
    
    valores_c1 = [1.0, 1.5, 2.0, 2.5, 3.0]
    valores_c2 = [3.0, 2.5, 2.0, 1.5, 1.0]
    valores_w = 0.6
    numero_particulas = [20, 30, 40, 60]
    num_execucoes_por_teste = 15

    resultados = []

    # VARIANDO O NÚMERO DE PARTÍCULAS
    for num_particulas in numero_particulas:
        for i in range(5):
            gbest_values = []
            avaliacoes_funcao = []
            sucessos = 0

            c1 = valores_c1[i]
            c2 = valores_c2[i]
            w = 0.6   

            for _ in range(num_execucoes_por_teste):
                dados_pso = calcular_pso(
                    num_particulas=num_particulas,
                    c1=c1,
                    c2=c2,
                    w=w,
                    max_iteracoes=100,
                    limites=(-500, 500),
                    capturar_posicoes=False
                )
                gbest_values.append(dados_pso['melhor_valor'])
                avaliacoes_funcao.append(dados_pso['numero_execucoes_funcao_objetivo'])
                sucesso = dados_pso.get('funcao_convergiu_para_minimo_esperado', False)
                if sucesso:
                    sucessos += 1

            media_gbest = np.mean(gbest_values)
            desvio_padrao_gbest = np.std(gbest_values)
            media_avaliacoes = np.mean(avaliacoes_funcao)
            taxa_sucesso = (sucessos / num_execucoes_por_teste) * 100

            resultados.append({
                'parametro': 'num_particulas',
                'valor': num_particulas,
                'media_gbest': round(media_gbest, 2),
                'desvio_padrao_gbest': round(desvio_padrao_gbest, 2),
                'media_avaliacoes_funcao': round(media_avaliacoes, 2),
                'taxa_sucesso_percentual': round(taxa_sucesso, 2),
                'valor c1': c1,
                'valor c2': c2,
                'valor w': w
            })

    # printando resultados em formato tabular:
    df_resultados = pd.DataFrame(resultados)
    print(df_resultados)
    # printando resultados em formato json 
    print(df_resultados.to_json(orient='records', indent=4))
    
testar_pso_metricas_completas()