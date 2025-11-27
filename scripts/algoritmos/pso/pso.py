from .Particula import Particula
import numpy as np

def calcular_pso(
    num_particulas=30,
    max_iteracoes=100,
    limites=(-500, 500),
    w=0.7,        # inércia
    c1=2,       # coeficiente cognitivo
    c2=2        # coeficiente social
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
    
    Retorna:
    - melhor_posicao: coordenadas (x, y) do mínimo encontrado
    - melhor_valor: valor mínimo encontrado
    - historico: lista com melhor valor em cada iteração
    """
    
    # Inicialização do enxame
    enxame = [Particula(limites) for _ in range(num_particulas)]
    
    # Inicialização da melhor posição global
    melhor_global_posicao = None
    melhor_global_valor = float('inf')
    
    # Histórico de convergência
    historico = []
    
    # Loop principal do PSO
    for iteracao in range(max_iteracoes):
        for particula in enxame:
            # Avaliar fitness
            valor_atual = particula.avaliar()
            
            # Atualizar melhor pessoal
            particula.atualizar_melhor(valor_atual)
            
            # Atualizar melhor global
            if valor_atual < melhor_global_valor:
                melhor_global_valor = valor_atual
                melhor_global_posicao = particula.posicao.copy()
        
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
        
        # Registrar histórico
        historico.append(melhor_global_valor)
        
        # Imprimir progresso a cada 10 iterações
        if (iteracao + 1) % 10 == 0:
            print(f"Iteração {iteracao + 1}/{max_iteracoes}: "
                  f"Melhor valor = {melhor_global_valor:.6f}")
    
    print("\nOtimização concluída!")
    print(f"Melhor posição encontrada: x = {melhor_global_posicao[0]:.4f}, "
          f"y = {melhor_global_posicao[1]:.4f}")
    print(f"Valor mínimo: {melhor_global_valor:.6f}")
    
    return melhor_global_posicao, melhor_global_valor, historico