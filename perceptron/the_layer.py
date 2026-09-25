"""
Autor       : Lucas Cruz
Data        : 23 de setembro 2026
Descrição   : Este código implementa uma classe Layer, apenas para fins didáticos
                de como ocorre a composição de vários neurônios.
Observações : 
    2. O Perceptron clássico é um modelo baseado em um único neurônio de decisão. 
    3. Uma camada pode ser formada por múltiplos neurônios trabalhando em paralelo.
"""
import numpy as np
from the_neuron import Neuron


class Layer():

    def __init__(self, n_inputs: int, n_neurons: int = 1, bias: int = -1):
        self.neurons = [
            Neuron(n_inputs, bias)
            for _ in range(n_neurons)
        ]

    def foward_pass(self, inputs: np.array):
        return [ neuron.foward_pass(inputs) for neuron in self.neurons ]