from scripts.plotagem import plotar_funcao_objetivo
from scripts.algoritmos import calcular_pso
from scripts.animacoes import criar_animacao_particulas

def app():
    print("Selecione opção desejada")
    opcao = input(
        "1 - Plotar função objetivo\n"
        "2 - Verificar dados através de PSO\n"
        "3 - Criar animação das partículas (2D)\n"
        "F - Sair\n"
    )

    if opcao == '1':
        return plotar_funcao_objetivo()
    elif opcao == '2':
        calcular_pso()
    elif opcao == '3':
        print("Criando animação 2D da movimentação das partículas...")
        return criar_animacao_particulas()

    return print("Encerrando aplicação.")
    
if __name__ == "__main__":
    app()