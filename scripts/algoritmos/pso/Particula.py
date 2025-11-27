import numpy as np
from scripts.funcoes_para_minizar import w18



class Particula:
    def __init__(self, limites):
        """Inicializa uma partícula com posição e velocidade aleatórias"""
        self.posicao = np.random.uniform(limites[0], limites[1], 2)
        self.velocidade = np.random.uniform(-1, 1, 2)
        self.melhor_posicao = self.posicao.copy()
        self.melhor_valor = float('inf')
    
    def avaliar(self):
        """Avalia a função objetivo na posição atual"""
        resultado = w18(self.posicao[0], self.posicao[1])
        # Garantir que retorna um escalar
        if isinstance(resultado, np.ndarray):
            return float(np.sum(resultado))  # ou np.mean(resultado) dependendo do contexto
        return float(resultado)
    
    def atualizar_melhor(self, valor_atual):
        """Atualiza a melhor posição pessoal da partícula"""
        if valor_atual < self.melhor_valor:
            self.melhor_valor = valor_atual
            self.melhor_posicao = self.posicao.copy()
        