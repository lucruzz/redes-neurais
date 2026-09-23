"""
Autor       : Lucas Cruz
Data        : 23 de setembro 2026
Descrição   : Este código implementa uma classe Perceptron, para representar 
                um neurônio com com uma função de decisão;
Observações : 
    1. A loss é calculada pelo MSE e é algo que podemos calcular para 
        avaliar o treinamento, mas ela não está sendo utilizada 
        diretamente para obter a atualização dos pesos. Isso será feito
        na implementação da MLP com backpropagation.
"""
import numpy as np
from the_neuron import Neuron


class Perceptron(Neuron):

    def __init__(self, n_inputs: int, bias: int = -1):
        super().__init__(n_inputs, bias)

    def train(
            self,
            x: np.array,
            y: float, 
            learning_rate: float = 0.01
        ) -> tuple[np.ndarray, np.ndarray]:

        # foward pass
        yhat = self.foward_pass(x)

        # cálculo do erro 
        delta = y - yhat

        # atualização dos pesos da entrada
        self.weights = self.weights + learning_rate * delta * x

        # atualização do bias
        self.bias = self.bias + learning_rate * delta

        return yhat


def main():
    # ===========================================================
    # Entradas: 
    #   - Vetor resposta
    #   - Matriz de entrada
    #   - Bias
    # ===========================================================

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
    # número máximo de épocas
    max_epochs = 500

    # vetor de saídas
    y_pred = np.array([])

    # número de épocas
    nepochs = 0

    # Inicializo um neurônio
    perceptron1 = Perceptron(n_inputs=X.shape[1])

    # critério de parada
    while not np.array_equal(Y, y_pred) and nepochs < max_epochs:

        tmp = np.array([])
        y_pred = np.array([])
        epoch_loss = 0

        for x, y in zip(X, Y):
            yhat = perceptron1.train(x, y)

            # acumula o erro quadrático
            epoch_loss += (y - yhat) ** 2

            tmp = np.append(tmp, yhat)
            
        y_pred = np.append(y_pred, tmp)

        # calcula a média dos erros
        epoch_loss /= len(X)

        nepochs += 1

        print(
            f'Época {nepochs} :: '
            f'y_pred: {y_pred} - '
            f'W: {perceptron1.weights} - '
            f'bias: {perceptron1.bias} - '
            f'loss: {epoch_loss}'
        )

    print(
        f'\n:: Término de execução da rede neural:\n'
        f'Época {nepochs} :: '
        f'y_pred: {y_pred} - '
        f'W: {perceptron1.weights} - '
        f'bias: {perceptron1.bias} - '
        f'loss: {epoch_loss}'
    )

if __name__ == '__main__':
    main()