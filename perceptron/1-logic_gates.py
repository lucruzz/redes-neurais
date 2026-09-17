"""
Autor       : Lucas Cruz
Data        : 9 de setembro 2026
Descrição   : Este código implementa um Perceptron de um único neurônio, 
              treinado para representar as portas lógicas AND e OR.
Observações : 
    1. Uma particularidade nesta implementação é que o limiar é fixo em 1.
    O que equivale a um bias fixo de valor -1. Significando que a rede
    aprende os pesos, mas não aprende o bias.

    2. Essa rede funciona para as portas lógicas AND e OR. No entanto,
    um único Perceptron não consegue representar a porta lógica XOR, 
    pois ela não é linearmente separável. Nesse caso, como o while 
    não tem limite de épocas, o treinamento não terminaria.
"""
import numpy as np

# ===========================================================
# Funções de apoio: 
#   - Função de ativação
#   - Função de processamento
#   - Função de atualização de pesos
# ===========================================================

def activation_fuction(u: int, z: int = 1) -> int:
    return 1 if u >= z else 0


def processing(inputs: np.array, weights: np.array) -> int:
    return np.dot(inputs, weights)


def update_weights(
        weights: np.array,
        delta: np.array, 
        x: np.array, 
        learning_rate: float
    ) -> np.array:

    return weights + (learning_rate * delta * x)

# ===========================================================
# Entradas: 
#   - Vetor resposta
#   - Matriz de entrada
# ===========================================================

# matriz de entradas (cada linha é um vetor de entradas)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# vetor resposta
# AND
Y = np.array([0, 0, 0, 1])
# OR
# Y = np.array([0, 1, 1, 1])

# ===========================================================
# Hiperparâmetros:
#   - Limiar de ativação (?)
#   - Taxa de aprendizado
# ===========================================================

# limiar de ativação
# z = 1

# taxa de aprendizado
lr = 0.3

# vetor de saídas
y_pred = np.array([])

# inicialização do vetor de pesos sinápticos
W = np.array([0, 0])

# número de épocas
nepochs = 0

# critério de parada
while not np.array_equal(Y, y_pred):

    tmp = np.array([])
    y_pred = np.array([])

    # passa por todo o conjunto de dados
    # X é a matriz de entrada
    # x é o vetor de entrada
    # Y é o vetor de saídas desejadas
    # y é uma amostra do vetor de saídas desejadas
    for x, y in zip(X, Y):

        # processa uma saída intermediária
        # 
        # x1 --> w1 - 
        #            \  /-------------------\
        #             > | v = x1*w1 + x2*w2 | -->
        #            /  \-------------------/
        # x2 --> w2 - 
        #
        v = processing(x, W)

        # aplica a função de ativação sobre a saída intermediária
        # 
        # v --> f(v) = y --> y
        tmp = np.append(tmp, [activation_fuction(v)])

        # calcula o erro 
        # pelo último valor da saída intermediária encontrada
        # 
        delta = y - tmp[-1]

        # atualização dos pesos
        W = update_weights(W, delta, x, lr)

    y_pred = np.append(y_pred, tmp)

    nepochs += 1
    print(f'Época {nepochs} :: y_pred: {y_pred} - W: {W}')

print(y_pred)