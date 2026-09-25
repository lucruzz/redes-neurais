"""
Autor       : Lucas Cruz
Data        : 25 de setembro 2026
Descrição   : Este código implementa uma classe Perceptron, para representar 
                um neurônio com com uma função de decisão;
Observações : 
    1. A loss é calculada pelo MSE e é algo que podemos calcular para 
        avaliar o treinamento, mas ela não está sendo utilizada 
        diretamente para obter a atualização dos pesos. Isso será feito
        na implementação da MLP com backpropagation.
"""
import numpy as np
from collections.abc import Callable


def mse(x: np.ndarray, y: np.ndarray):
    # a função np.mean já soma os elementos e divide pela quantidade deles
    return np.mean( (x - y) ** 2 )


def dmse(x: np.ndarray, y: np.ndarray):
    return 2 * (x - y) / len(x) 


def sigmoid(u: np.ndarray) -> np.ndarray:
    return 1 / ( 1 + np.exp(-u) ) # <- s(x) = s_x


def dsigmoid(s_x: np.ndarray) -> np.ndarray:
    return ( 1 - s_x ) * s_x


class Neuron:
    def __init__(self, n_inputs: int, bias: int = -1, activation_fuction: Callable = sigmoid):
        self.weights = np.random.rand(n_inputs)
        self.bias = bias
        self.activation_function = activation_fuction

    def foward_pass(self, inputs: np.ndarray):
        v = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(v)


class Layer():

    def __init__(self, n_inputs: int, n_neurons: int = 1, activation_function: Callable = sigmoid, bias: int = -1):
        self.neurons = [
            Neuron(n_inputs, bias, activation_function)
            for _ in range(n_neurons)
        ]

    def foward_pass(self, inputs: np.ndarray):
        return [ neuron.foward_pass(inputs) for neuron in self.neurons ]


class MultiLayerPerceptron():

    def __init__(self, n_inputs: int, n_layers: list = [2, 1], activations_functions: list = [sigmoid, sigmoid], bias: int = -1):

        self.layers = []

        current_layer = n_inputs

        for n_neurons, activation_function in zip(n_layers, activations_functions):
            self.layers.append( Layer(n_inputs=current_layer, n_neurons=n_neurons, activation_function=activation_function, bias=bias) )
            current_layer = n_neurons


    def foward(self, x: np.ndarray) -> list:
        
        current_input = x
        intermediate_outputs = []
        
        for layer in self.layers:
            tmp = layer.foward_pass(current_input)
            current_input = tmp
            intermediate_outputs.append(tmp)
        
        return intermediate_outputs

    
    def backpropagation(self, x: np.ndarray, Z: list, dE_dz: np.ndarray) -> list:

        nlayers = len(self.layers)
        
        current_gradients = dE_dz
        gradients = [None] * nlayers
        
        for i in range(nlayers -1, -1, -1):
            layer = self.layers[i]

            # pego as saídas após a função de ativação da camada atual
            outputs = np.asarray(Z[i])

            # pego agora as entradas que foram recebidas pela camada atual
            inputs = np.asarray(
                x if i == 0 else Z[i - 1]
            )

            # pego os pesos dos neurônios da camada atual
            # para calcular os gradientes posteriormente
            W = np.array([ neuron.weights for neuron in layer.neurons ])

            # o delta aqui é a derivada da loss em relação aos valores 
            # antes da função de ativação
            delta = current_gradients * dsigmoid(outputs)

            # derivada da loss em relação aos pesos
            dE_dw = np.outer(delta, inputs)
            dE_db = delta.copy()

            gradients[i] = (dE_dw, dE_db)

            # calculo os gratientes atuais
            current_gradients = W.T @ delta

        return gradients


    def gradient_descendent(self, gradients: list, learning_rate: float = 0.01) -> None:
        for layer, (dE_dw, dE_db) in zip(self.layers, gradients):
            for j, neuron in enumerate(layer.neurons):
                neuron.weights -= learning_rate * dE_dw[j]
                neuron.bias -= learning_rate * dE_db[j]
        

    def train(
            self,
            x: np.ndarray,
            y: np.ndarray, 
            learning_rate: float = 0.01,
            loss_function: Callable = mse,
            dloss_function: Callable = dmse
        ) -> tuple[np.ndarray, np.ndarray]:

        # foward pass
        Z = self.foward(x)

        # backward pass

        # calcular o loss usando a loss function
        yhat_outputs = Z[-1]
        loss = loss_function(yhat_outputs, y)
        
        # calcular a derivada do Erro em relação às saídas finais
        dE_dz = dloss_function(yhat_outputs, y)

        # calcular o gradiente
        gradients = self.backpropagation(x, Z, dE_dz)

        # agora é só fazer a atualização dos pesos e dos biases
        # lembrando que isso é a descida do gradiente
        self.gradient_descendent(gradients, learning_rate)

        return yhat_outputs, loss


def main():
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

    # vetor resposta: porta lógica XOR
    Y = np.array([0, 1, 1, 0])

    # configuro a semente para reprodução
    np.random.seed(42)

    # número máximo de épocas
    max_epochs = 10000

    # vetor de saídas
    y_pred = np.array([])

    # número de épocas
    nepochs = 0

    lr = 0.5

    # Inicializo um neurônio
    mlp = MultiLayerPerceptron(n_inputs=X.shape[1], n_layers=[3, 2, 1], activations_functions=[sigmoid,sigmoid,sigmoid])

    # critério de parada
    while not np.array_equal(Y, y_pred) and nepochs < max_epochs:

        tmp = np.array([])
        y_pred = np.array([])
        epoch_loss = 0

        for x, y in zip(X, Y):
            yhat, loss = mlp.train(x, y, learning_rate=lr)

            # acumula o erro
            epoch_loss += loss

            tmp = np.append(tmp, yhat)
            
        y_pred_activation_function = np.append(y_pred, tmp)
        
        y_pred = (y_pred_activation_function >= 0.5).astype(int)

        # calcula a média dos erros
        epoch_loss /= len(X)

        nepochs += 1

        print(
            f'Época {nepochs} :: '
            f'y_pred: {y_pred} - '
            f'loss: {epoch_loss}'
        )

    print(
        f'\n:: Término de execução da rede neural:\n'
        f'Época {nepochs} :: '
        f'y_pred: {y_pred} - '
        f'loss: {epoch_loss}'
    )

if __name__ == '__main__':
    main()