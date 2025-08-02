def calcular_sensibilidade(VP, FN):
 
    return VP / (VP + FN)

def calcular_especificidade(VN, FP):

    return VN / (FP + VN)

def calcular_acuracia(VP, VN, FP, FN):

    N = VP + VN + FP + FN
    return (VP + VN) / N

def calcular_precisao(VP, FP):

    return VP / (VP + FP)

def calcular_f_score(precisao, sensibilidade):

    return 2 * (precisao * sensibilidade) / (precisao + sensibilidade)

def main():
  
    VP = 50
    VN = 40
    FP = 10
    FN = 5


    sensibilidade = calcular_sensibilidade(VP, FN)
    especificidade = calcular_especificidade(VN, FP)
    acuracia = calcular_acuracia(VP, VN, FP, FN)
    precisao = calcular_precisao(VP, FP)
    f_score = calcular_f_score(precisao, sensibilidade)

    
    print(f"Sensibilidade (Recall): {sensibilidade:.2f}")
    print(f"Especificidade: {especificidade:.2f}")
    print(f"Acurácia: {acuracia:.2f}")
    print(f"Precisão: {precisao:.2f}")
    print(f"F-score: {f_score:.2f}")

if __name__ == "__main__":
    main()
