
import numpy as np
import random
from scripts.funcoes_para_minizar import w18
from constants import constantes_ag, minimo_funcao

def calcular_algoritmo_genetico(
    numero_individuos=constantes_ag.NUMERO_INDIVIDUOS,
    taxa_crossover=constantes_ag.TAXA_CROSSOVER,  # Percentual dos mais aptos que irão cruzar
    taxa_mutacao=constantes_ag.TAXA_MUTACAO,    # Percentual dos indivíduos que irão sofrer mutação
    numero_geracoes=constantes_ag.NUMERO_GERACOES, # Número de gerações que o algoritmo irá rodar (critério de parada)
    limites=constantes_ag.LIMITES_ESPACO_BUSCA,
    intensidade_mutacao=constantes_ag.INTENSIDADE_MUTACAO,
    geracoes_sem_melhoria=constantes_ag.GERACOES_SEM_MELHORIA,  # Número de gerações sem melhoria para parar
    salvar_posicoes=False,
):
    """
    Executa o algoritmo genético para otimizar a função W18.
    
    Returns:
        - object: dicionário com os resultados do algoritmo
    """
    
    # 1. Criar população inicial
    populacao = criar_populacao_inicial(numero_individuos, limites)
    
    # Histórico para acompanhar convergência
    historico_fitness = []
    melhor_global = None
    melhor_fitness_global = float('-inf') # o float('-inf') representa o menor valor possível
    contador_sem_melhoria = 0
    posicoes_populacao = [] if salvar_posicoes else None
    
    # Avaliar população inicial
    melhor_da_geracao = max(populacao, key=lambda x: x[1])
    valor_w18_inicial = -melhor_da_geracao[1]  # Converter fitness de volta para valor W18
    historico_fitness.append(valor_w18_inicial)
    numero_execucoes_funcao_objetivo = 0
    
    if melhor_da_geracao[1] > melhor_fitness_global:
        melhor_fitness_global = melhor_da_geracao[1]
        melhor_global = melhor_da_geracao[0].copy()
    
    # Loop principal das gerações
    for geracao in range(numero_geracoes):
        nova_populacao = []
        
        # Manter os melhores (elitismo - 10%)
        populacao_ordenada = sorted(populacao, key=lambda x: x[1], reverse=True)
        num_elites = max(2, int(0.1 * numero_individuos)) # Pelo menos 2 elites, mas pode ser 10% da população
        for i in range(num_elites):
            nova_populacao.append(populacao_ordenada[i])
        
        # Gerar o resto da população através de crossover e mutação
        while len(nova_populacao) < numero_individuos:
            # Seleção de pais
            pai1 = selecao_torneio(populacao)
            pai2 = selecao_torneio(populacao)
            
            # Crossover
            if random.random() < taxa_crossover: 
                filho1, filho2 = crossover_blx_alfa(pai1, pai2)
            else:
                filho1, filho2 = pai1.copy(), pai2.copy()
            
            # Mutação
            filho1 = mutacao_uniforme(filho1, taxa_mutacao, intensidade_mutacao)
            filho2 = mutacao_uniforme(filho2, taxa_mutacao, intensidade_mutacao)
            
            # Avaliar fitness dos filhos
            fitness1 = avaliar_fitness(filho1)
            fitness2 = avaliar_fitness(filho2)
            numero_execucoes_funcao_objetivo += 2
            
            # Adicionar à nova população
            nova_populacao.append((filho1, fitness1))
            if len(nova_populacao) < numero_individuos:
                nova_populacao.append((filho2, fitness2))
        
        # Substituir população
        populacao = nova_populacao[:numero_individuos]
        
        # Encontrar melhor da geração
        melhor_da_geracao = max(populacao, key=lambda x: x[1])
        valor_w18_atual = -melhor_da_geracao[1]
        historico_fitness.append(valor_w18_atual)
        
        # Atualizar melhor global e verificar melhoria
        melhorou = False
        print(melhor_da_geracao)
        print(melhor_fitness_global)
        if round(melhor_da_geracao[1], 2) > round(melhor_fitness_global, 2):
            melhorou = True
            contador_sem_melhoria = 0
            melhor_fitness_global = melhor_da_geracao[1]
            melhor_global = melhor_da_geracao[0].copy()
        
        # Incrementar contador se não houve melhoria
        if not melhorou:
            contador_sem_melhoria += 1
        
        # Verificando critério de parada por convergência
        if contador_sem_melhoria >= geracoes_sem_melhoria:
            break

        # Salvar posições da população atual
        if salvar_posicoes:
            posicoes_atual = [individuo[0].copy() for individuo in populacao]
            posicoes_populacao.append(posicoes_atual)
    
    # Resultado final
    melhor_valor_w18 = -melhor_fitness_global
    
    if contador_sem_melhoria >= geracoes_sem_melhoria:
        print(f"Motivo da parada: Convergência (sem melhoria por {geracoes_sem_melhoria} gerações)")
    else:
        print("Motivo da parada: Número máximo de gerações atingido")

    funcao_convergiu_para_minimo_esperado = False
    diferenca_para_minimo = abs(melhor_valor_w18 - minimo_funcao.MINIMO_FUNCAO_W18)
    if diferenca_para_minimo <= minimo_funcao.TOLERANCIA_DIVERGENCIA:
        funcao_convergiu_para_minimo_esperado = True
    
    dados = {
        'melhor_posicao': melhor_global,
        'melhor_valor': melhor_valor_w18,
        'historico_fitness': historico_fitness,
        "quantidade_geracoes_realizadas": len(historico_fitness),
        "numero_individuos": numero_individuos,
        "taxa_crossover": taxa_crossover,
        "taxa_mutacao": taxa_mutacao,
        "intensidade_mutacao": intensidade_mutacao,
        "geracoes_sem_melhoria": geracoes_sem_melhoria,
        "numero_geracoes": numero_geracoes,
        "posicoes_populacao": posicoes_populacao,
        "numero_execucoes_funcao_objetivo": numero_execucoes_funcao_objetivo,
        "funcao_convergiu_para_minimo_esperado": funcao_convergiu_para_minimo_esperado
    }

    exibir_dados_ag(dados)

    return dados

def criar_individuo(limites=(-500, 500)):
    return np.random.uniform(limites[0], limites[1], 2) # o random.uniform retorna um array numpy de 2 elementos com valores entre os limites

def avaliar_fitness(individuo):
    """Avalia o fitness do indivíduo usando W18."""
    resultado = w18(individuo[0], individuo[1])
    valor = np.min(resultado)  # o np.min pega o menor valor de um array numpy
    return -valor

def criar_populacao_inicial(numero_individuos, limites=(-500, 500)):
    populacao = []
    for i in range(numero_individuos):
        individuo = criar_individuo(limites)
        fitness = avaliar_fitness(individuo)
        populacao.append((individuo, fitness))
    
    print(f"População inicial criada com {len(populacao)} indivíduos")
    return populacao

def selecao_torneio(populacao, tamanho_torneio=3):
    candidatos = random.sample(populacao, min(tamanho_torneio, len(populacao))) # Seleciona aleatoriamente candidatos para o torneio
    melhor = max(candidatos, key=lambda x: x[1]) # Seleciona o indivíduo com melhor fitness
    return melhor[0]

def crossover_blx_alfa(pai1, pai2, alfa=0.5): 
    """Crossover BLX-α entre dois pais."""
    filho1 = np.zeros(2) # Criar arrays numpy vazios para os filhos
    filho2 = np.zeros(2)

    for i in range(2):
        print(f"Pai1 gene {i}: {pai1[i]}, Pai2 gene {i}: {pai2[i]}")
        # Calcular limites do intervalo
        min_val = min(pai1[i], pai2[i]) 
        max_val = max(pai1[i], pai2[i])
        intervalo = max_val - min_val
        
        # Expandir intervalo
        limite_inf = min_val - alfa * intervalo
        limite_sup = max_val + alfa * intervalo
        
        # Gerar filhos no intervalo expandido
        filho1[i] = np.random.uniform(limite_inf, limite_sup)
        filho2[i] = np.random.uniform(limite_inf, limite_sup)
    
    # Manter nos limites do problema
    filho1 = np.clip(filho1, -500, 500)
    filho2 = np.clip(filho2, -500, 500)
    
    return filho1, filho2

def mutacao_uniforme(individuo, taxa_mutacao, intensidade=10.0): # a intensidade define o quanto a mutação pode alterar o gene
    """Aplica mutação uniforme ao indivíduo."""
    individuo_mutado = individuo.copy()
    
    for i in range(len(individuo_mutado)): # Para cada gene do indivíduo
        if random.random() < taxa_mutacao:
            # Adicionar perturbação uniforme
            perturbacao = np.random.uniform(-intensidade, intensidade)
            individuo_mutado[i] += perturbacao
            # Manter nos limites
            individuo_mutado[i] = np.clip(individuo_mutado[i], -500, 500)
    
    return individuo_mutado

def exibir_dados_ag(dados):
    melhor_posicao = dados['melhor_posicao']
    melhor_valor = dados['melhor_valor']
    quantidade_geracoes_realizadas = dados['quantidade_geracoes_realizadas']
    numero_individuos = dados['numero_individuos']
    taxa_crossover = dados['taxa_crossover']
    taxa_mutacao = dados['taxa_mutacao']
    intensidade_mutacao = dados['intensidade_mutacao']
    geracoes_sem_melhoria = dados['geracoes_sem_melhoria']
    numero_geracoes = dados['numero_geracoes']
    numero_execucoes_funcao_objetivo = dados['numero_execucoes_funcao_objetivo']

    print("\n" + "="*70)
    print(" "*20 + "RESULTADOS DO ALGORITMO GENÉTICO")
    print("="*70)
    print(f"{'Métrica':<40} {'Valor':>28}")
    print("-"*70)
    print(f"{'Melhor posição X':<40} {round(melhor_posicao[0], 2):>28}")
    print(f"{'Melhor posição Y':<40} {round(melhor_posicao[1], 2):>28}")
    print(f"{'Valor mínimo encontrado':<40} {round(melhor_valor, 2):>28}")
    print(f"{'Iterações realizadas':<40} {quantidade_geracoes_realizadas:>28}")
    print(f"{'Número de execuções da função objetivo':<40} {numero_execucoes_funcao_objetivo:>28}")
    print(f"{'Convergiu para mínimo esperado':<40} {str(dados.get('funcao_convergiu_para_minimo_esperado', False)):>28}")
    print("-"*70)
    print(f"{'Parametros do PSO':<40} {'Valor':>28}")
    print("-"*70)
    print(f"{'Número de individuos':<40} {numero_individuos:>28}")
    print(f"{'Taxa de Crossover':<40} {taxa_crossover:>28.2f}")
    print(f"{'Taxa de Mutação':<40} {taxa_mutacao:>28.2f}")
    print(f"{'Intensidade da Mutação':<40} {intensidade_mutacao:>28.2f}")
    print(f"{'Gerações sem Melhoria':<40} {geracoes_sem_melhoria:>28}")
    print(f"{'Número de Gerações':<40} {numero_geracoes:>28}")
    print("="*70 + "\n")
