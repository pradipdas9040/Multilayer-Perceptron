from typing import Tuple
from common import np


class AdaGrad:
    def __init__(self, learning_rate):
        """
        AdaGrad
        :param learning_rate:
        """

        self._learning_rate = learning_rate

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

        raise 