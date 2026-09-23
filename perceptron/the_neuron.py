"""
Autor       : Lucas Cruz
Data        : 23 de setembro 2026
Descrição   : Este código implementa:
                1. Uma classe Neuron, para representar o neurônio;
Observações : 
    1. Um neurônio pode ser entendido como uma unidade de processamento;
    2. Essencialmente, esse script apenas muda "as coisas" de lugar;
"""
import numpy as np


class Neuron:
    def __init__(self, n_inputs: int, bias: int = -1):
        self.weights = np.random.rand(n_inputs)
        self.bias = bias

    def _activation_function(self, u: np.array):
        return np.where(u >= 0, 1, 0)

    def foward_pass(self, inputs: np.array):
        v = np.dot(inputs, self.weights) + self.bias
        return self._activation_function(v)


# ===========================================================
# Funções de apoio: 
#   - Função de atualização de pesos
# ===========================================================
def update_weights(
        weights: np.array,
        delta: np.array, 
        x: np.array, 
        learning_rate: float
    ) -> np.ndarray:

    return weights + (learning_rate * delta * x)


def main():
    # ===========================================================
    # Entradas: 
    #   - Vetor resposta
    #   - Matriz de entrada
    #   - Bias
    # ===========================================================
    bias = -1

    # matriz de entradas (cada linha é um vetor de entradas)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # vetor resposta
    # OR
    Y = np.array([0, 1, 1, 1])

    # configuro a semente para reprodução
    np.random.seed(42)

    # ===========================================================
    # Hiperparâmetros:
    #   - Taxa de aprendizado
    # ===========================================================

    # taxa de aprendizado
    lr = 0.3

    # vetor de saídas
    y_pred = np.array([])

    # número de épocas
    nepochs = 0

    # Inicializo um neurônio
    neuron1 = Neuron(n_inputs=X.shape[1])

    # critério de parada
    while not np.array_equal(Y, y_pred):

        tmp = np.array([])
        y_pred = np.array([])

        for x, y in zip(X, Y):

            tmp = np.append(tmp, neuron1.foward_pass(x))
            
            # calcula o erro 
            # pelo último valor da saída intermediária encontrada
            delta = y - tmp[-1]

            # atualização dos pesos da entrada
            neuron1.weights = update_weights(neuron1.weights, delta, x, lr)

            # atualização do bias
            neuron1.bias = neuron1.bias + lr * delta

        y_pred = np.append(y_pred, tmp)

        nepochs += 1
        print(f'Época {nepochs} :: y_pred: {y_pred} - W: {neuron1.weights} - bias: {neuron1.bias}')

    print(y_pred)

if __name__=='__main__':
    main()