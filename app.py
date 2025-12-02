from scripts.plotagem import plotar_funcao_objetivo
from scripts.algoritmos import calcular_pso, calcular_algoritmo_genetico
from scripts.animacoes import criar_animacao_particulas

def app():
    print("Selecione opção desejada")
    opcao = input(
        "1 - Plotar função objetivo\n"
        "2 - Verificar dados através de PSO\n"
        "3 - Verificar dados através de Algoritmo Genético\n"
        "4 - Criar animação das partículas (PSO)\n"
        "5 - Criar animação das populações (AG)\n"
        "F - Sair\n"
    )

    if opcao == '1':
        return plotar_funcao_objetivo()
    elif opcao == '2':
        calcular_pso()
    elif opcao == '3':
        calcular_algoritmo_genetico()
    elif opcao == '4':
        print("Criando animação 2D da movimentação das partículas...")
        return criar_animacao_particulas()
    elif opcao == '5':
        print("Criando animação 2D da evolução das populações...")
        return criar_animacao_particulas(algoritmo_selecionado='ag')

    return print("Encerrando aplicação.")
    
if __name__ == "__main__":
    app()