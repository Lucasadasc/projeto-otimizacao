import numpy as np
import pandas as pd
import itertools
import time
from datetime import datetime
import json
from scripts.algoritmos import calcular_algoritmo_genetico

# ================================================================================
# 📊 ANÁLISE DOS RESULTADOS - ALGORITMO GENÉTICO
# ================================================================================

# 🏆 TOP 5 - MELHOR VALOR MÉDIO:
#   722. Ind: 70, cross:0.9, mut:0.10, intens: 5.0, ger_sem:12 → -1046.6987 (±0.0001)
#   895. Ind:100, cross:0.7, mut:0.20, intens:20.0, ger_sem:16 → -1046.6986 (±0.0001)
#   685. Ind: 70, cross:0.8, mut:0.15, intens:20.0, ger_sem: 8 → -1046.6985 (±0.0002)
#   759. Ind: 70, cross:0.9, mut:0.20, intens:10.0, ger_sem:16 → -1046.6985 (±0.0003)
#   990. Ind:100, cross:0.9, mut:0.10, intens:20.0, ger_sem:12 → -1046.6985 (±0.0004)

# 🎯 TOP 5 - MELHOR VALOR ABSOLUTO:
#   404. Ind: 50, cross:0.8, mut:0.10, intens: 5.0, ger_sem:20 → -1046.6988
#   300. Ind: 50, cross:0.6, mut:0.15, intens:15.0, ger_sem:20 → -1046.6988
#   484. Ind: 50, cross:0.9, mut:0.15, intens: 5.0, ger_sem:20 → -1046.6988
#   44. Ind: 30, cross:0.6, mut:0.15, intens:15.0, ger_sem:20 → -1046.6988
#   526. Ind: 70, cross:0.6, mut:0.05, intens:20.0, ger_sem:12 → -1046.6988

# 🎲 TOP 5 - MAIOR CONSISTÊNCIA (menor desvio padrão):
#   696. Ind: 70, cross:0.8, mut:0.20, intens:10.0, ger_sem:20 → ±0.0001 (média: -1046.6968)
#   286. Ind: 50, cross:0.6, mut:0.10, intens:20.0, ger_sem:12 → ±0.0001 (média: -1046.6969)
#   722. Ind: 70, cross:0.9, mut:0.10, intens: 5.0, ger_sem:12 → ±0.0001 (média: -1046.6987)
#   620. Ind: 70, cross:0.7, mut:0.15, intens:15.0, ger_sem:20 → ±0.0001 (média: -1046.6971)
#   895. Ind:100, cross:0.7, mut:0.20, intens:20.0, ger_sem:16 → ±0.0001 (média: -1046.6986)

# ⚡ TOP 5 - MAIOR EFICIÊNCIA (menos gerações):
#   737. Ind: 70, cross:0.9, mut:0.15, intens: 5.0, ger_sem: 8 → 17.0 ger (valor: -840.1566)
#   265. Ind: 50, cross:0.6, mut:0.05, intens:15.0, ger_sem: 8 → 17.7 ger (valor: -875.8236)
#   977. Ind:100, cross:0.9, mut:0.10, intens: 5.0, ger_sem: 8 → 17.7 ger (valor: -876.1978)
#   157. Ind: 30, cross:0.8, mut:0.10, intens:20.0, ger_sem: 8 → 18.3 ger (valor: -723.3669)
#   769. Ind:100, cross:0.6, mut:0.05, intens: 5.0, ger_sem: 8 → 18.3 ger (valor: -1046.6972)

# 💡 TOP 5 - MELHOR EFICIÊNCIA POPULACIONAL:
#   44. Ind: 30, cross:0.6, mut:0.15, intens:15.0, ger_sem:20 → -34.8900 (valor/indivíduo)
#   142. Ind: 30, cross:0.8, mut:0.05, intens:20.0, ger_sem:12 → -34.8900 (valor/indivíduo)
#   210. Ind: 30, cross:0.9, mut:0.10, intens: 5.0, ger_sem:12 → -34.8900 (valor/indivíduo)
#   175. Ind: 30, cross:0.8, mut:0.15, intens:20.0, ger_sem:16 → -34.8900 (valor/indivíduo)
#   184. Ind: 30, cross:0.8, mut:0.20, intens:10.0, ger_sem:20 → -34.8900 (valor/indivíduo)

# 📈 ANÁLISE POR PARÂMETRO:

#    NUMERO INDIVIDUOS:
#      100: -1018.9302
#      70: -970.6823
#      50: -907.1417
#      30: -813.5035

#    TAXA CROSSOVER:
#      0.8: -931.3523
#      0.9: -930.6459
#      0.6: -924.9042
#      0.7: -923.3554

#    TAXA MUTACAO:
#      0.2: -945.5347
#      0.15: -926.4763
#      0.05: -920.4641
#      0.1: -917.7827

#    INTENSIDADE MUTACAO:
#      15.0: -937.5077
#      20.0: -925.2043
#      10.0: -924.6853
#      5.0: -922.8605

#    GERACOES SEM MELHORIA:
#      8: -938.6274
#      12: -930.9328
#      20: -925.8120
#      16: -914.8855

# 🏅 CONFIGURAÇÃO RECOMENDADA:
#    Número de Indivíduos: 30
#    Taxa de Crossover: 0.8
#    Taxa de Mutação: 0.1
#    Intensidade de Mutação: 10.0
#    Gerações sem Melhoria: 8
#    → Valor médio: -1046.6913 (±0.0095)
#    → Score composto: 0.961

def testar_parametros_ag():
    """
    Testa diferentes combinações de parâmetros do Algoritmo Genético para encontrar a configuração ótima.
    
    Parâmetros testados:
    - numero_individuos: tamanho da população
    - taxa_crossover: probabilidade de cruzamento
    - taxa_mutacao: probabilidade de mutação
    - intensidade_mutacao: magnitude das mutações
    - geracoes_sem_melhoria: critério de parada por estagnação
    """
    
    print("🧬 INICIANDO TESTE SISTEMÁTICO DE PARÂMETROS AG")
    print("="*80)
    
    # Definir ranges de parâmetros para testar
    parametros_teste = {
        'numero_individuos': [30, 50, 70, 100],
        'taxa_crossover': [0.6, 0.7, 0.8, 0.9],
        'taxa_mutacao': [0.05, 0.1, 0.15, 0.2],
        'intensidade_mutacao': [5.0, 10.0, 15.0, 20.0],
        'geracoes_sem_melhoria': [8, 12, 16, 20]
    }
    
    # Parâmetros fixos
    numero_geracoes = 40  # Reduzido para acelerar testes
    limites = (-500, 500)
    tolerancia_melhoria = 1e-6
    num_execucoes = 3  # Número de execuções por combinação para média
    
    print("📊 Configuração do teste:")
    print(f"   • Indivíduos: {parametros_teste['numero_individuos']}")
    print(f"   • Taxa Crossover: {parametros_teste['taxa_crossover']}")
    print(f"   • Taxa Mutação: {parametros_teste['taxa_mutacao']}")
    print(f"   • Intensidade Mutação: {parametros_teste['intensidade_mutacao']}")
    print(f"   • Gerações sem melhoria: {parametros_teste['geracoes_sem_melhoria']}")
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
        num_individuos, taxa_cross, taxa_mut, intens_mut, ger_sem_melhoria = combo
        
        print(f"\n🧪 Testando combinação {i+1}/{len(combinacoes)}")
        print(f"   Indivíduos: {num_individuos}, cross: {taxa_cross}, mut: {taxa_mut}, "
              f"intens: {intens_mut}, ger_sem: {ger_sem_melhoria}")
        
        # Executar múltiplas vezes para obter média
        valores_obtidos = []
        geracoes_realizadas = []
        tempos_execucao = []
        
        for execucao in range(num_execucoes):
            inicio_exec = time.time()
            
            # Executar AG com os parâmetros atuais
            resultado = calcular_algoritmo_genetico(
                numero_individuos=num_individuos,
                taxa_crossover=taxa_cross,
                taxa_mutacao=taxa_mut,
                numero_geracoes=numero_geracoes,
                limites=limites,
                intensidade_mutacao=intens_mut,
                geracoes_sem_melhoria=ger_sem_melhoria,
                tolerancia_melhoria=tolerancia_melhoria,
                salvar_posicoes=False
            )
            
            fim_exec = time.time()
            
            valores_obtidos.append(resultado["melhor_valor"])
            geracoes_realizadas.append(resultado["quantidade_geracoes_realizadas"])
            tempos_execucao.append(fim_exec - inicio_exec)
            
            print(f"     Exec {execucao+1}: {resultado['melhor_valor']:.4f} "
                  f"({resultado['quantidade_geracoes_realizadas']} ger)")
        
        # Calcular estatísticas
        media_valor = np.mean(valores_obtidos)
        std_valor = np.std(valores_obtidos)
        melhor_valor = np.min(valores_obtidos)
        pior_valor = np.max(valores_obtidos)
        media_geracoes = np.mean(geracoes_realizadas)
        media_tempo = np.mean(tempos_execucao)
        
        # Armazenar resultado
        resultado_combo = {
            'numero_individuos': num_individuos,
            'taxa_crossover': taxa_cross,
            'taxa_mutacao': taxa_mut,
            'intensidade_mutacao': intens_mut,
            'geracoes_sem_melhoria': ger_sem_melhoria,
            'media_valor': media_valor,
            'std_valor': std_valor,
            'melhor_valor': melhor_valor,
            'pior_valor': pior_valor,
            'media_geracoes': media_geracoes,
            'media_tempo': media_tempo,
            'consistencia': 1.0 / (1.0 + std_valor),  # Métrica de consistência
            'eficiencia_populacional': melhor_valor / num_individuos,  # Qualidade por indivíduo
            'eficiencia_temporal': melhor_valor / media_tempo  # Qualidade por segundo
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
    print("📊 ANÁLISE DOS RESULTADOS - ALGORITMO GENÉTICO")
    print("="*80)
    
    # Top 5 melhores configurações por diferentes critérios
    print("\n🏆 TOP 5 - MELHOR VALOR MÉDIO:")
    top_media = df_resultados.nsmallest(5, 'media_valor')
    for i, row in top_media.iterrows():
        print(f"  {i+1}. Ind:{int(row['numero_individuos']):3d}, cross:{row['taxa_crossover']:.1f}, "
              f"mut:{row['taxa_mutacao']:.2f}, intens:{row['intensidade_mutacao']:4.1f}, "
              f"ger_sem:{int(row['geracoes_sem_melhoria']):2d} "
              f"→ {row['media_valor']:.4f} (±{row['std_valor']:.4f})")
    
    print("\n🎯 TOP 5 - MELHOR VALOR ABSOLUTO:")
    top_melhor = df_resultados.nsmallest(5, 'melhor_valor')
    for i, row in top_melhor.iterrows():
        print(f"  {i+1}. Ind:{int(row['numero_individuos']):3d}, cross:{row['taxa_crossover']:.1f}, "
              f"mut:{row['taxa_mutacao']:.2f}, intens:{row['intensidade_mutacao']:4.1f}, "
              f"ger_sem:{int(row['geracoes_sem_melhoria']):2d} "
              f"→ {row['melhor_valor']:.4f}")
    
    print("\n🎲 TOP 5 - MAIOR CONSISTÊNCIA (menor desvio padrão):")
    top_consistencia = df_resultados.nlargest(5, 'consistencia')
    for i, row in top_consistencia.iterrows():
        print(f"  {i+1}. Ind:{int(row['numero_individuos']):3d}, cross:{row['taxa_crossover']:.1f}, "
              f"mut:{row['taxa_mutacao']:.2f}, intens:{row['intensidade_mutacao']:4.1f}, "
              f"ger_sem:{int(row['geracoes_sem_melhoria']):2d} "
              f"→ ±{row['std_valor']:.4f} (média: {row['media_valor']:.4f})")
    
    print("\n⚡ TOP 5 - MAIOR EFICIÊNCIA (menos gerações):")
    top_eficiencia = df_resultados.nsmallest(5, 'media_geracoes')
    for i, row in top_eficiencia.iterrows():
        print(f"  {i+1}. Ind:{int(row['numero_individuos']):3d}, cross:{row['taxa_crossover']:.1f}, "
              f"mut:{row['taxa_mutacao']:.2f}, intens:{row['intensidade_mutacao']:4.1f}, "
              f"ger_sem:{int(row['geracoes_sem_melhoria']):2d} "
              f"→ {row['media_geracoes']:.1f} ger (valor: {row['media_valor']:.4f})")
    
    print("\n💡 TOP 5 - MELHOR EFICIÊNCIA POPULACIONAL:")
    top_efic_pop = df_resultados.nsmallest(5, 'eficiencia_populacional')
    for i, row in top_efic_pop.iterrows():
        print(f"  {i+1}. Ind:{int(row['numero_individuos']):3d}, cross:{row['taxa_crossover']:.1f}, "
              f"mut:{row['taxa_mutacao']:.2f}, intens:{row['intensidade_mutacao']:4.1f}, "
              f"ger_sem:{int(row['geracoes_sem_melhoria']):2d} "
              f"→ {row['eficiencia_populacional']:.4f} (valor/indivíduo)")
    
    # Análise estatística por parâmetro
    print("\n📈 ANÁLISE POR PARÂMETRO:")
    for param in ['numero_individuos', 'taxa_crossover', 'taxa_mutacao', 
                  'intensidade_mutacao', 'geracoes_sem_melhoria']:
        media_por_param = df_resultados.groupby(param)['media_valor'].mean().sort_values()
        print(f"\n   {param.upper().replace('_', ' ')}:")
        for valor, media in media_por_param.items():
            print(f"     {valor}: {media:.4f}")
    
    # Encontrar configuração ótima (combinando múltiplos critérios)
    print("\n🏅 CONFIGURAÇÃO RECOMENDADA:")
    
    # Normalizar métricas (0-1) e criar score composto
    df_norm = df_resultados.copy()
    df_norm['norm_valor'] = 1 - (df_norm['media_valor'] - df_norm['media_valor'].min()) / (df_norm['media_valor'].max() - df_norm['media_valor'].min())
    df_norm['norm_consistencia'] = (df_norm['consistencia'] - df_norm['consistencia'].min()) / (df_norm['consistencia'].max() - df_norm['consistencia'].min())
    df_norm['norm_eficiencia'] = 1 - (df_norm['media_geracoes'] - df_norm['media_geracoes'].min()) / (df_norm['media_geracoes'].max() - df_norm['media_geracoes'].min())
    df_norm['norm_efic_pop'] = 1 - (df_norm['eficiencia_populacional'] - df_norm['eficiencia_populacional'].min()) / (df_norm['eficiencia_populacional'].max() - df_norm['eficiencia_populacional'].min())
    
    # Score composto (pesos ajustáveis)
    peso_valor = 0.5
    peso_consistencia = 0.2
    peso_eficiencia = 0.15
    peso_efic_pop = 0.15
    
    df_norm['score_composto'] = (peso_valor * df_norm['norm_valor'] + 
                                 peso_consistencia * df_norm['norm_consistencia'] + 
                                 peso_eficiencia * df_norm['norm_eficiencia'] +
                                 peso_efic_pop * df_norm['norm_efic_pop'])
    
    melhor_config = df_norm.loc[df_norm['score_composto'].idxmax()]
    
    print(f"   Número de Indivíduos: {int(melhor_config['numero_individuos'])}")
    print(f"   Taxa de Crossover: {melhor_config['taxa_crossover']}")
    print(f"   Taxa de Mutação: {melhor_config['taxa_mutacao']}")
    print(f"   Intensidade de Mutação: {melhor_config['intensidade_mutacao']}")
    print(f"   Gerações sem Melhoria: {int(melhor_config['geracoes_sem_melhoria'])}")
    print(f"   → Valor médio: {melhor_config['media_valor']:.4f} (±{melhor_config['std_valor']:.4f})")
    print(f"   → Score composto: {melhor_config['score_composto']:.3f}")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar CSV
    nome_csv = f"resultados_teste_ag_{timestamp}.csv"
    df_resultados.to_csv(nome_csv, index=False)
    print(f"\n💾 Resultados salvos em: {nome_csv}")
    
    # Salvar configuração recomendada
    config_recomendada = {
        'numero_individuos': int(melhor_config['numero_individuos']),
        'taxa_crossover': float(melhor_config['taxa_crossover']),
        'taxa_mutacao': float(melhor_config['taxa_mutacao']),
        'intensidade_mutacao': float(melhor_config['intensidade_mutacao']),
        'geracoes_sem_melhoria': int(melhor_config['geracoes_sem_melhoria']),
        'tolerancia_melhoria': 1e-6,
        'performance': {
            'media_valor': float(melhor_config['media_valor']),
            'std_valor': float(melhor_config['std_valor']),
            'media_geracoes': float(melhor_config['media_geracoes']),
            'eficiencia_populacional': float(melhor_config['eficiencia_populacional']),
            'score_composto': float(melhor_config['score_composto'])
        },
        'data_teste': timestamp,
        'total_combinacoes_testadas': len(combinacoes),
        'execucoes_por_combinacao': num_execucoes
    }
    
    nome_json = f"config_ag_otima_{timestamp}.json"
    with open(nome_json, 'w', encoding='utf-8') as f:
        json.dump(config_recomendada, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Configuração ótima salva em: {nome_json}")
    
    print("\n" + "="*80)
    print("✅ TESTE CONCLUÍDO COM SUCESSO!")
    print("="*80)
    
    return df_resultados, config_recomendada

def testar_configuracao_ag_rapido():
    """
    Teste rápido com menos combinações para desenvolvimento/debugging.
    """
    print("🚀 TESTE RÁPIDO DE PARÂMETROS AG")
    print("="*60)
    
    # Configurações reduzidas para teste rápido
    parametros_teste = {
        'numero_individuos': [30, 50],
        'taxa_crossover': [0.7, 0.8],
        'taxa_mutacao': [0.1, 0.15],
        'intensidade_mutacao': [10.0, 15.0],
        'geracoes_sem_melhoria': [10, 15]
    }
    
    # Executar teste com parâmetros reduzidos
    df_resultados, melhor_config = testar_parametros_ag_customizado(
        parametros_teste, 
        num_execucoes=2, 
        numero_geracoes=25
    )
    
    print("\n✅ TESTE RÁPIDO CONCLUÍDO!")
    print("Melhor configuração encontrada:")
    print(f"  Indivíduos: {int(melhor_config['numero_individuos'])}")
    print(f"  Cross: {melhor_config['taxa_crossover']}, Mut: {melhor_config['taxa_mutacao']}")
    print(f"  Intens: {melhor_config['intensidade_mutacao']}")
    print(f"  Valor médio: {melhor_config['media_valor']:.4f}")
    
    return df_resultados, melhor_config

def testar_parametros_ag_customizado(parametros_teste, num_execucoes=3, numero_geracoes=40):
    """
    Versão customizável do teste de parâmetros do AG.
    """
    print("🔬 INICIANDO TESTE CUSTOMIZADO DE PARÂMETROS AG")
    print("="*60)
    
    # Parâmetros fixos
    limites = (-500, 500)
    tolerancia_melhoria = 1e-6
    
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
        num_individuos, taxa_cross, taxa_mut, intens_mut, ger_sem_melhoria = combo
        
        print(f"\n🧪 Testando combinação {i+1}/{len(combinacoes)}")
        print(f"   Ind: {num_individuos}, cross: {taxa_cross}, mut: {taxa_mut}, "
              f"intens: {intens_mut}, ger_sem: {ger_sem_melhoria}")
        
        # Executar múltiplas vezes para obter média
        valores_obtidos = []
        geracoes_realizadas = []
        
        for execucao in range(num_execucoes):
            resultado = calcular_algoritmo_genetico(
                numero_individuos=num_individuos,
                taxa_crossover=taxa_cross,
                taxa_mutacao=taxa_mut,
                numero_geracoes=numero_geracoes,
                limites=limites,
                intensidade_mutacao=intens_mut,
                geracoes_sem_melhoria=ger_sem_melhoria,
                tolerancia_melhoria=tolerancia_melhoria,
                salvar_posicoes=False
            )
            
            valores_obtidos.append(resultado["melhor_valor"])
            geracoes_realizadas.append(resultado["quantidade_geracoes_realizadas"])
            
            print(f"     Exec {execucao+1}: {resultado['melhor_valor']:.4f}")
        
        # Calcular estatísticas
        media_valor = np.mean(valores_obtidos)
        std_valor = np.std(valores_obtidos)
        melhor_valor = np.min(valores_obtidos)
        media_geracoes = np.mean(geracoes_realizadas)
        
        # Armazenar resultado
        resultado_combo = {
            'numero_individuos': num_individuos,
            'taxa_crossover': taxa_cross,
            'taxa_mutacao': taxa_mut,
            'intensidade_mutacao': intens_mut,
            'geracoes_sem_melhoria': ger_sem_melhoria,
            'media_valor': media_valor,
            'std_valor': std_valor,
            'melhor_valor': melhor_valor,
            'media_geracoes': media_geracoes
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
        print(f"  {i+1}. Ind:{int(row['numero_individuos']):2d}, cross:{row['taxa_crossover']:.1f}, "
              f"mut:{row['taxa_mutacao']:.2f}, intens:{row['intensidade_mutacao']:4.1f} "
              f"→ {row['media_valor']:.4f} (±{row['std_valor']:.4f})")
    
    return df_resultados, top_3.iloc[0].to_dict()

if __name__ == "__main__":
    print("Escolha o tipo de teste para o Algoritmo Genético:")
    print("1 - Teste completo (demorado, mas abrangente)")
    print("2 - Teste rápido (menos combinações)")
    print("3 - Sair")
    
    escolha = input("Digite sua escolha (1-3): ").strip()
    
    if escolha == "1":
        resultados, config_otima = testar_parametros_ag()
    elif escolha == "2":
        resultados, config_otima = testar_configuracao_ag_rapido()
    elif escolha == "3":
        print("Saindo...")
    else:
        print("Opção inválida!")