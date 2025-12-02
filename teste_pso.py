import numpy as np
import pandas as pd
import itertools
import time
from datetime import datetime
import json
from scripts.algoritmos import calcular_pso

def testar_parametros_pso():
    """
    Testa diferentes combinações de parâmetros do PSO para encontrar a configuração ótima.
    
    Parâmetros testados:
    - num_particulas: número de partículas no enxame
    - w: peso de inércia
    - c1: coeficiente cognitivo
    - c2: coeficiente social
    - max_iteracoes_sem_melhoria: critério de parada por estagnação
    """
    
    print("🔬 INICIANDO TESTE SISTEMÁTICO DE PARÂMETROS PSO")
    print("="*80)
    
    # Definir ranges de parâmetros para testar
    parametros_teste = {
        'num_particulas': [10, 15, 20, 25, 30],
        'w': [0.4, 0.6, 0.8, 1.0, 1.2],  # peso de inércia
        'c1': [1.0, 1.5, 2.0, 2.5],  # coeficiente cognitivo
        'c2': [1.0, 1.5, 2.0, 2.5],  # coeficiente social
        'max_iteracoes_sem_melhoria': [10, 15, 20, 25]
    }
    
    # Parâmetros fixos
    max_iteracoes = 50  # Reduzido para acelerar testes
    limites = (-500, 500)
    num_execucoes = 3  # Número de execuções por combinação para média
    
    print("📊 Configuração do teste:")
    print(f"   • Partículas: {parametros_teste['num_particulas']}")
    print(f"   • Peso de Inércia (w): {parametros_teste['w']}")
    print(f"   • Coef. Cognitivo (c1): {parametros_teste['c1']}")
    print(f"   • Coef. Social (c2): {parametros_teste['c2']}")
    print(f"   • Iter. sem melhoria: {parametros_teste['max_iteracoes_sem_melhoria']}")
    print(f"   • Execuções por combo: {num_execucoes}")
    
    # Calcular total de combinações
    total_combos = 1
    for param in parametros_teste.values():
        total_combos *= len(param)
    
    print(f"   • Total de combinações: {total_combos}")
    print(f"   • Total de execuções: {total_combos * num_execucoes}")
    print("="*80)
    
    # Gerar todas as combinações possíveis
    valores_param = list(parametros_teste.values())
    combinacoes = list(itertools.product(*valores_param))
    
    resultados = []
    inicio_teste = time.time()
    
    print(f"⏳ Iniciando teste de {len(combinacoes)} combinações...")
    
    for i, combo in enumerate(combinacoes):
        num_particulas, w, c1, c2, max_iter_sem_melhoria = combo
        
        print(f"\n🧪 Testando combinação {i+1}/{len(combinacoes)}")
        print(f"   Partículas: {num_particulas}, w: {w}, c1: {c1}, c2: {c2}, "
              f"iter_sem_melhoria: {max_iter_sem_melhoria}")
        
        # Executar múltiplas vezes para obter média
        valores_obtidos = []
        iteracoes_realizadas = []
        tempos_execucao = []
        
        for execucao in range(num_execucoes):
            inicio_exec = time.time()
            
            # Executar PSO com os parâmetros atuais
            resultado = calcular_pso(
                num_particulas=num_particulas,
                max_iteracoes=max_iteracoes,
                limites=limites,
                w=w,
                c1=c1,
                c2=c2,
                capturar_posicoes=False,
                max_iteracoes_sem_melhoria=max_iter_sem_melhoria
            )
            
            fim_exec = time.time()
            
            valores_obtidos.append(resultado["melhor_valor"])
            iteracoes_realizadas.append(resultado["quantidade_iteracoes_realizadas"])
            tempos_execucao.append(fim_exec - inicio_exec)
            
            print(f"     Exec {execucao+1}: {resultado['melhor_valor']:.4f} "
                  f"({resultado['quantidade_iteracoes_realizadas']} iter)")
        
        # Calcular estatísticas
        media_valor = np.mean(valores_obtidos)
        std_valor = np.std(valores_obtidos)
        melhor_valor = np.min(valores_obtidos)
        pior_valor = np.max(valores_obtidos)
        media_iteracoes = np.mean(iteracoes_realizadas)
        media_tempo = np.mean(tempos_execucao)
        
        # Armazenar resultado
        resultado_combo = {
            'num_particulas': num_particulas,
            'w': w,
            'c1': c1,
            'c2': c2,
            'max_iter_sem_melhoria': max_iter_sem_melhoria,
            'media_valor': media_valor,
            'std_valor': std_valor,
            'melhor_valor': melhor_valor,
            'pior_valor': pior_valor,
            'media_iteracoes': media_iteracoes,
            'media_tempo': media_tempo,
            'consistencia': 1.0 / (1.0 + std_valor)  # Métrica de consistência
        }
        
        resultados.append(resultado_combo)
        
        print(f"   📈 Média: {media_valor:.4f} (±{std_valor:.4f})")
        print(f"   🎯 Melhor: {melhor_valor:.4f}, Pior: {pior_valor:.4f}")
    
    fim_teste = time.time()
    tempo_total = fim_teste - inicio_teste
    
    print(f"\n⏰ Teste concluído em {tempo_total:.2f} segundos ({tempo_total/60:.1f} min)")
    
    # Converter para DataFrame para análise
    df_resultados = pd.DataFrame(resultados)
    
    # Análise dos resultados
    print("\n" + "="*80)
    print("📊 ANÁLISE DOS RESULTADOS")
    print("="*80)
    
    # Top 10 melhores configurações por diferentes critérios
    print("\n🏆 TOP 5 - MELHOR VALOR MÉDIO:")
    top_media = df_resultados.nsmallest(5, 'media_valor')
    for i, row in top_media.iterrows():
        print(f"  {i+1}. Partículas:{int(row['num_particulas']):2d}, w:{row['w']:.1f}, "
              f"c1:{row['c1']:.1f}, c2:{row['c2']:.1f}, iter_sem:{int(row['max_iter_sem_melhoria']):2d} "
              f"→ {row['media_valor']:.4f} (±{row['std_valor']:.4f})")
    
    print("\n🎯 TOP 5 - MELHOR VALOR ABSOLUTO:")
    top_melhor = df_resultados.nsmallest(5, 'melhor_valor')
    for i, row in top_melhor.iterrows():
        print(f"  {i+1}. Partículas:{int(row['num_particulas']):2d}, w:{row['w']:.1f}, "
              f"c1:{row['c1']:.1f}, c2:{row['c2']:.1f}, iter_sem:{int(row['max_iter_sem_melhoria']):2d} "
              f"→ {row['melhor_valor']:.4f}")
    
    print("\n🎲 TOP 5 - MAIOR CONSISTÊNCIA (menor desvio padrão):")
    top_consistencia = df_resultados.nlargest(5, 'consistencia')
    for i, row in top_consistencia.iterrows():
        print(f"  {i+1}. Partículas:{int(row['num_particulas']):2d}, w:{row['w']:.1f}, "
              f"c1:{row['c1']:.1f}, c2:{row['c2']:.1f}, iter_sem:{int(row['max_iter_sem_melhoria']):2d} "
              f"→ ±{row['std_valor']:.4f} (média: {row['media_valor']:.4f})")
    
    print("\n⚡ TOP 5 - MAIOR EFICIÊNCIA (menos iterações):")
    top_eficiencia = df_resultados.nsmallest(5, 'media_iteracoes')
    for i, row in top_eficiencia.iterrows():
        print(f"  {i+1}. Partículas:{int(row['num_particulas']):2d}, w:{row['w']:.1f}, "
              f"c1:{row['c1']:.1f}, c2:{row['c2']:.1f}, iter_sem:{int(row['max_iter_sem_melhoria']):2d} "
              f"→ {row['media_iteracoes']:.1f} iter (valor: {row['media_valor']:.4f})")
    
    # Análise estatística por parâmetro
    print("\n📈 ANÁLISE POR PARÂMETRO:")
    for param in ['num_particulas', 'w', 'c1', 'c2', 'max_iter_sem_melhoria']:
        media_por_param = df_resultados.groupby(param)['media_valor'].mean().sort_values()
        print(f"\n   {param.upper()}:")
        for valor, media in media_por_param.items():
            print(f"     {valor}: {media:.4f}")
    
    # Encontrar configuração ótima (combinando múltiplos critérios)
    print("\n🏅 CONFIGURAÇÃO RECOMENDADA:")
    
    # Normalizar métricas (0-1) e criar score composto
    df_norm = df_resultados.copy()
    df_norm['norm_valor'] = 1 - (df_norm['media_valor'] - df_norm['media_valor'].min()) / (df_norm['media_valor'].max() - df_norm['media_valor'].min())
    df_norm['norm_consistencia'] = (df_norm['consistencia'] - df_norm['consistencia'].min()) / (df_norm['consistencia'].max() - df_norm['consistencia'].min())
    df_norm['norm_eficiencia'] = 1 - (df_norm['media_iteracoes'] - df_norm['media_iteracoes'].min()) / (df_norm['media_iteracoes'].max() - df_norm['media_iteracoes'].min())
    
    # Score composto (pesos ajustáveis)
    peso_valor = 0.6
    peso_consistencia = 0.25
    peso_eficiencia = 0.15
    
    df_norm['score_composto'] = (peso_valor * df_norm['norm_valor'] + 
                                 peso_consistencia * df_norm['norm_consistencia'] + 
                                 peso_eficiencia * df_norm['norm_eficiencia'])
    
    melhor_config = df_norm.loc[df_norm['score_composto'].idxmax()]
    
    print(f"   Partículas: {int(melhor_config['num_particulas'])}")
    print(f"   Peso de Inércia (w): {melhor_config['w']}")
    print(f"   Coef. Cognitivo (c1): {melhor_config['c1']}")
    print(f"   Coef. Social (c2): {melhor_config['c2']}")
    print(f"   Max Iter sem Melhoria: {int(melhor_config['max_iter_sem_melhoria'])}")
    print(f"   → Valor médio: {melhor_config['media_valor']:.4f} (±{melhor_config['std_valor']:.4f})")
    print(f"   → Score composto: {melhor_config['score_composto']:.3f}")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar CSV
    nome_csv = f"resultados_teste_pso_{timestamp}.csv"
    df_resultados.to_csv(nome_csv, index=False)
    print(f"\n💾 Resultados salvos em: {nome_csv}")
    
    # Salvar configuração recomendada
    config_recomendada = {
        'num_particulas': int(melhor_config['num_particulas']),
        'w': float(melhor_config['w']),
        'c1': float(melhor_config['c1']),
        'c2': float(melhor_config['c2']),
        'max_iter_sem_melhoria': int(melhor_config['max_iter_sem_melhoria']),
        'performance': {
            'media_valor': float(melhor_config['media_valor']),
            'std_valor': float(melhor_config['std_valor']),
            'media_iteracoes': float(melhor_config['media_iteracoes']),
            'score_composto': float(melhor_config['score_composto'])
        },
        'data_teste': timestamp,
        'total_combinacoes_testadas': len(combinacoes),
        'execucoes_por_combinacao': num_execucoes
    }
    
    nome_json = f"config_pso_otima_{timestamp}.json"
    with open(nome_json, 'w', encoding='utf-8') as f:
        json.dump(config_recomendada, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Configuração ótima salva em: {nome_json}")
    
    print("\n" + "="*80)
    print("✅ TESTE CONCLUÍDO COM SUCESSO!")
    print("="*80)
    
    return df_resultados, config_recomendada

def testar_configuracao_rapido():
    """
    Teste rápido com menos combinações para desenvolvimento/debugging.
    """
    print("🚀 TESTE RÁPIDO DE PARÂMETROS PSO")
    print("="*60)
    
    # Configurações reduzidas para teste rápido
    parametros_teste = {
        'num_particulas': [15, 25],
        'w': [0.6, 0.8, 1.0],
        'c1': [1.5, 2.0],
        'c2': [1.5, 2.0],
        'max_iteracoes_sem_melhoria': [15, 20]
    }
    
    # Executar teste com parâmetros reduzidos
    df_resultados, melhor_config = testar_parametros_pso_customizado(
        parametros_teste, 
        num_execucoes=2, 
        max_iteracoes=30
    )
    
    print("\n✅ TESTE RÁPIDO CONCLUÍDO!")
    print("Melhor configuração encontrada:")
    print(f"  Partículas: {int(melhor_config['num_particulas'])}")
    print(f"  w: {melhor_config['w']}, c1: {melhor_config['c1']}, c2: {melhor_config['c2']}")
    print(f"  Valor médio: {melhor_config['media_valor']:.4f}")
    
    return df_resultados, melhor_config

def testar_parametros_pso_customizado(parametros_teste, num_execucoes=3, max_iteracoes=50):
    """
    Versão customizável do teste de parâmetros.
    """
    print("🔬 INICIANDO TESTE CUSTOMIZADO DE PARÂMETROS PSO")
    print("="*60)
    
    # Parâmetros fixos
    limites = (-500, 500)
    
    print("📊 Configuração do teste:")
    for param, valores in parametros_teste.items():
        print(f"   • {param}: {valores}")
    print(f"   • Execuções por combo: {num_execucoes}")
    
    # Calcular total de combinações
    total_combos = 1
    for param in parametros_teste.values():
        total_combos *= len(param)
    
    print(f"   • Total de combinações: {total_combos}")
    print(f"   • Total de execuções: {total_combos * num_execucoes}")
    print("="*60)
    
    # Gerar todas as combinações possíveis
    valores_param = list(parametros_teste.values())
    combinacoes = list(itertools.product(*valores_param))
    
    resultados = []
    inicio_teste = time.time()
    
    print(f"⏳ Iniciando teste de {len(combinacoes)} combinações...")
    
    for i, combo in enumerate(combinacoes):
        num_particulas, w, c1, c2, max_iter_sem_melhoria = combo
        
        print(f"\n🧪 Testando combinação {i+1}/{len(combinacoes)}")
        print(f"   Partículas: {num_particulas}, w: {w}, c1: {c1}, c2: {c2}, "
              f"iter_sem_melhoria: {max_iter_sem_melhoria}")
        
        # Executar múltiplas vezes para obter média
        valores_obtidos = []
        iteracoes_realizadas = []
        
        for execucao in range(num_execucoes):
            resultado = calcular_pso(
                num_particulas=num_particulas,
                max_iteracoes=max_iteracoes,
                limites=limites,
                w=w,
                c1=c1,
                c2=c2,
                capturar_posicoes=False,
                max_iteracoes_sem_melhoria=max_iter_sem_melhoria
            )
            
            valores_obtidos.append(resultado["melhor_valor"])
            iteracoes_realizadas.append(resultado["quantidade_iteracoes_realizadas"])
            
            print(f"     Exec {execucao+1}: {resultado['melhor_valor']:.4f}")
        
        # Calcular estatísticas
        media_valor = np.mean(valores_obtidos)
        std_valor = np.std(valores_obtidos)
        melhor_valor = np.min(valores_obtidos)
        media_iteracoes = np.mean(iteracoes_realizadas)
        
        # Armazenar resultado
        resultado_combo = {
            'num_particulas': num_particulas,
            'w': w,
            'c1': c1,
            'c2': c2,
            'max_iter_sem_melhoria': max_iter_sem_melhoria,
            'media_valor': media_valor,
            'std_valor': std_valor,
            'melhor_valor': melhor_valor,
            'media_iteracoes': media_iteracoes
        }
        
        resultados.append(resultado_combo)
        print(f"   📈 Média: {media_valor:.4f} (±{std_valor:.4f})")
    
    fim_teste = time.time()
    tempo_total = fim_teste - inicio_teste
    
    print(f"\n⏰ Teste concluído em {tempo_total:.2f} segundos")
    
    # Converter para DataFrame
    df_resultados = pd.DataFrame(resultados)
    
    # Mostrar top 3 melhores configurações
    print("\n🏆 TOP 3 - MELHORES CONFIGURAÇÕES:")
    top_3 = df_resultados.nsmallest(3, 'media_valor')
    for i, row in top_3.iterrows():
        print(f"  {i+1}. Partículas:{int(row['num_particulas']):2d}, w:{row['w']:.1f}, "
              f"c1:{row['c1']:.1f}, c2:{row['c2']:.1f} "
              f"→ {row['media_valor']:.4f} (±{row['std_valor']:.4f})")
    
    return df_resultados, top_3.iloc[0].to_dict()

if __name__ == "__main__":
    print("Escolha o tipo de teste:")
    print("1 - Teste completo (demorado, mas abrangente)")
    print("2 - Teste rápido (menos combinações)")
    print("3 - Sair")
    
    escolha = input("Digite sua escolha (1-3): ").strip()
    
    if escolha == "1":
        resultados, config_otima = testar_parametros_pso()
    elif escolha == "2":
        resultados, config_otima = testar_configuracao_rapido()
    elif escolha == "3":
        print("Saindo...")
    else:
        print("Opção inválida!")

#  Teste concluído em 170.14 segundos (2.8 mi
# ================================================================================
# 📊 ANÁLISE DOS RESULTADOS
# # ==============================================================================
# 🏆 TOP 5 - MELHOR VALOR MÉDIO:
#   321. Partículas:15, w:0.4, c1:1.0, c2:1.0, iter_sem:10 → -1046.6988 (±0.0000)
#   324. Partículas:15, w:0.4, c1:1.0, c2:1.0, iter_sem:25 → -1046.6988 (±0.0000)
#   642. Partículas:20, w:0.4, c1:1.0, c2:1.0, iter_sem:15 → -1046.6988 (±0.0000)
#   963. Partículas:25, w:0.4, c1:1.0, c2:1.0, iter_sem:20 → -1046.6988 (±0.0000)
# #   1287. Partículas:30, w:0.4, c1:1.0, c2:1.5, iter_sem:20 → -1046.6988 (±0.000
# 🎯 TOP 5 - MELHOR VALOR ABSOLUTO:
#   323. Partículas:15, w:0.4, c1:1.0, c2:1.0, iter_sem:20 → -1046.6988
#   324. Partículas:15, w:0.4, c1:1.0, c2:1.0, iter_sem:25 → -1046.6988
#   641. Partículas:20, w:0.4, c1:1.0, c2:1.0, iter_sem:10 → -1046.6988
#   1284. Partículas:30, w:0.4, c1:1.0, c2:1.0, iter_sem:25 → -1046.6988
# #   1. Partículas:10, w:0.4, c1:1.0, c2:1.0, iter_sem:10 → -1046.69
# 🎲 TOP 5 - MAIOR CONSISTÊNCIA (menor desvio padrão):
#   321. Partículas:15, w:0.4, c1:1.0, c2:1.0, iter_sem:10 → ±0.0000 (média: -1046.6988)
#   642. Partículas:20, w:0.4, c1:1.0, c2:1.0, iter_sem:15 → ±0.0000 (média: -1046.6988)
#   963. Partículas:25, w:0.4, c1:1.0, c2:1.0, iter_sem:20 → ±0.0000 (média: -1046.6988)
#   1297. Partículas:30, w:0.4, c1:1.5, c2:1.0, iter_sem:10 → ±0.0000 (média: -1046.6988)
# #   324. Partículas:15, w:0.4, c1:1.0, c2:1.0, iter_sem:25 → ±0.0000 (média: -1046.698
# ⚡ TOP 5 - MAIOR EFICIÊNCIA (menos iterações):
#   565. Partículas:15, w:1.0, c1:2.5, c2:1.5, iter_sem:10 → 13.0 iter (valor: -817.4807)
#   1541. Partículas:30, w:1.2, c1:1.0, c2:1.5, iter_sem:10 → 14.0 iter (valor: -917.0140)
#   317. Partículas:10, w:1.2, c1:2.5, c2:2.5, iter_sem:10 → 14.3 iter (valor: -760.2251)
#   1581. Partículas:30, w:1.2, c1:2.0, c2:2.5, iter_sem:10 → 14.3 iter (valor: -960.5076)
# #   269. Partículas:10, w:1.2, c1:1.0, c2:2.5, iter_sem:10 → 14.7 iter (valor: -858.742
# # 📈 ANÁLISE POR PARÂMETR
#    NUM_PARTICULAS:
    #  30: -963.7356
    #  25: -934.4506
    #  20: -907.6040
    #  15: -875.9330
    # #  10: -822.47
#    W:
    #  0.8: -939.0973
    #  1.0: -928.5178
    #  0.6: -888.9862
    #  1.2: -876.9571
    # #  0.4: -870.64
#    C1:
    #  2.5: -924.7195
    #  2.0: -908.0750
    #  1.5: -897.8919
    # #  1.0: -872.67
#    C2:
    #  2.0: -912.4567
    #  2.5: -908.3205
    #  1.5: -899.3914
    # #  1.0: -883.19
#    MAX_ITER_SEM_MELHORIA:
    #  25: -912.2093
    #  20: -905.1889
    #  15: -899.9996
    # #  10: -885.96
# 🏅 CONFIGURAÇÃO RECOMENDADA:
#    Partículas: 25
#    Peso de Inércia (w): 0.6
#    Coef. Cognitivo (c1): 2.5
#    Coef. Social (c2): 2.5
#    Max Iter sem Melhoria: 10
#    → Valor médio: -1044.8004 (±0.2736)
# #    → Score composto: 0.9
# 💾 Resultados salvos em: resultados_teste_pso_20251201_171717.csv
# 💾 Configuração ótima salva em: config_pso_otima_20251201_171717.json
