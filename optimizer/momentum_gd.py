from typing import Tuple
from common import np


class MomentumGradientDescent:
    def __init__(self, learning_rate, momentum_factor):
        """
        :param learning_rate:
        :param momentum_factor
        """

        self._learning_rate = learning_rate
        self._momentum_factor = momentum_factor

        self._weight_updates = None  # most recent weight update produced
        self._bias_updates = None  # most recent bias update produced

    @property
    def weight_updates(self):
        # most recent weight update produced
        return self._weight_updates

    @property
    def bias_updates(self):
        # most recent bias update produced
        return self._bias_updates

    def calculate_update(self, gradw: np.ndarray, gradb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the amount of weight and bias updates to be made
        (to be subtracted)
        :param gradw: Gradient matrix of loss wrt weights
        :param gradb: Gradient matrix of loss wrt biases
        :return: Weight and bias updates to be made (subtracted)
        """

        # initialise weight and bias update history
        if self._weight_updates is None:
            self._weight_updates = np.zeros(gradw.shape)
            self._bias_updates = np.zeros(gradb.shape)

        self._weight_updates = self._momentum_factor * self._weight_updates + self._learning_rate * gradw
        self._bias_updates = self._momentum_factor * self._bias_updates + self._learning_rate * gradb

        return self._weight_updates, self._bias_updates