from .Particula import Particula
import numpy as np 

def calcular_pso(
    num_particulas=30,
    max_iteracoes=100,
    limites=(-500, 500),
    w=0.9,        # inércia
    c1=1.9,       # coeficiente cognitivo
    c2=1.9,       # coeficiente social
    capturar_posicoes=False,  # capturar posições das partículas para visualização
    max_iteracoes_sem_melhoria=30  # critério de parada por estagnação
):
    """
    Implementa o algoritmo PSO para minimizar a função W18.
    
    Parâmetros:
    - num_particulas: número de partículas no enxame
    - max_iteracoes: número máximo de iterações
    - limites: tupla (min, max) definindo o espaço de busca
    - w: peso de inércia (controla exploração vs exploração)
    - c1: coeficiente cognitivo (atração pela melhor posição pessoal)
    - c2: coeficiente social (atração pela melhor posição global)
    - max_iteracoes_sem_melhoria: número de iterações sem melhoria para parar
    
    Retorna:
    - melhor_posicao: coordenadas (x, y) do mínimo encontrado
    - melhor_valor: valor mínimo encontrado
    - historico: lista com melhor valor em cada iteração
    - posicoes_particulas: (se capturar_posicoes=True) lista das posições de todas as partículas em cada iteração
    """
    
    # Inicialização do enxame
    enxame = [Particula(limites) for _ in range(num_particulas)]
    
    # Inicialização da melhor posição global
    melhor_global_posicao = None
    melhor_global_valor = float('inf')
    quantidade_iteracoes_realizadas = 0
    
    # Histórico de convergência
    historico = []
    
    # Critério de parada por estagnação
    iteracoes_sem_melhoria = 0
    melhor_valor_anterior = float('inf')
    
    # Capturar posições das partículas (se solicitado)
    posicoes_particulas = [] if capturar_posicoes else None
    
    # Loop principal do PSO
    for iteracao in range(max_iteracoes):
        quantidade_iteracoes_realizadas += 1

        for particula in enxame:
            # Avaliar fitness
            valor_atual = particula.avaliar()
            
            # Atualizar melhor pessoal
            particula.atualizar_melhor(valor_atual)
            
            # Atualizar melhor global
            if valor_atual < melhor_global_valor:
                melhor_global_valor = valor_atual
                melhor_global_posicao = particula.posicao.copy()
                iteracoes_sem_melhoria = 0  # Resetar contador
        
        # Atualizar velocidades e posições
        for particula in enxame:
            r1 = np.random.random(2)
            r2 = np.random.random(2)
            
            # Componente cognitivo (atração pela melhor posição pessoal)
            cognitivo = c1 * r1 * (particula.melhor_posicao - particula.posicao)
            
            # Componente social (atração pela melhor posição global)
            social = c2 * r2 * (melhor_global_posicao - particula.posicao)
            
            # Atualizar velocidade
            particula.velocidade = w * particula.velocidade + cognitivo + social
            
            # Atualizar posição
            particula.posicao = particula.posicao + particula.velocidade
            
            # Aplicar limites de busca
            particula.posicao = np.clip(particula.posicao, limites[0], limites[1])
        
        # Capturar posições das partículas (se solicitado)
        if capturar_posicoes:
            posicoes_iteracao = [particula.posicao.copy() for particula in enxame]
            posicoes_particulas.append(posicoes_iteracao)
        
        # Registrar histórico
        historico.append(melhor_global_valor)
        
        # Verificar se houve melhoria nesta iteração
        if melhor_global_valor >= melhor_valor_anterior:
            iteracoes_sem_melhoria += 1
        melhor_valor_anterior = melhor_global_valor
        
        # Critério de parada por estagnação
        if iteracoes_sem_melhoria >= max_iteracoes_sem_melhoria:
            print(f"\nParada antecipada na iteração {iteracao + 1}: "
                  f"Sem melhoria por {max_iteracoes_sem_melhoria} iterações consecutivas")
            break

    dados = {
        "melhor_posicao": melhor_global_posicao,
        "melhor_valor": melhor_global_valor,
        "historico": historico,
        "quantidade_iteracoes_realizadas": quantidade_iteracoes_realizadas
    }
    
    if capturar_posicoes:
        dados["posicoes_particulas"] = posicoes_particulas

    exibir_dados_pso(dados, c1, c2, w)
    
    return dados

def exibir_dados_pso(dados_pso, c1, c2, w):
    melhor_posicao = dados_pso["melhor_posicao"]
    melhor_valor = dados_pso["melhor_valor"]
    quantidade_iteracoes_realizadas = dados_pso["quantidade_iteracoes_realizadas"]
    
    print("\n" + "="*70)
    print(" "*28 + "RESULTADOS DO PSO")
    print("="*70)
    print(f"{'Métrica':<40} {'Valor':>28}")
    print("-"*70)
    print(f"{'Melhor posição X':<40} {round(melhor_posicao[0], 2):>28}")
    print(f"{'Melhor posição Y':<40} {round(melhor_posicao[1], 2):>28}")
    print(f"{'Valor mínimo encontrado':<40} {round(melhor_valor, 2):>28}")
    print(f"{'Iterações realizadas':<40} {quantidade_iteracoes_realizadas:>28}")
    print("-"*70)
    print(f"{'Parametros do PSO':<40} {'Valor':>28}")
    print("-"*70)
    print(f"{'Coeficiente Cognitivo (c1)':<40} {c1:>28.2f}")
    print(f"{'Coeficiente Social (c2)':<40} {c2:>28.2f}")
    print(f"{'Peso de Inércia (w)':<40} {w:>28.2f}")
    print("="*70 + "\n")

