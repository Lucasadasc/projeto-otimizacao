from scripts.plotagem import plotar_funcao_objetivo
from scripts.algoritmos import calcular_pso

def app():
    print("Selecione opção desejada")
    opcao = input("1 - Plotar função objetivo\n2 - Verificar dados através de PSO \n3 - Sair\n")

    if opcao == '1':
        return plotar_funcao_objetivo()
    elif opcao == '2':
        dados, valor, historico = calcular_pso()
        return (f"Melhor posição: {dados}, Valor da função objetivo: {valor}")

    return print("Encerrando aplicação.")
    
if __name__ == "__main__":
    app()