from common import np
from typing import Tuple
from constants import Defaults


class Adam:
    def __init__(self, beta1=Defaults.beta1, beta2=Defaults.beta2, learning_rate=Defaults.learning_rate):
        """
        Adam optimizer.
        Time step is incremented internally every time calculate_update method is called.
        :param beta1: 1st moment decay rate
        :param beta2: 2nd moment decay rate
        :param learning_rate: Learning rate
        """

        self._beta1 = beta1
        self._beta2 = beta2
        self._learning_rate = learning_rate
        self._eps = Defaults.eps

        # for weights
        self._m_w = None
        self._v_w = None

        # for bias
        self._m_b = None
        self._v_b = None

        self._weight_updates = None  # most recent weight update produced
        self._bias_updates = None  # most recent bias update produced

        self._time_step = 1

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

        if self._m_w is None:
            self._m_w = np.zeros(gradw.shape)
            self._v_w = np.zeros(gradw.shape)
            self._m_b = np.zeros(gradb.shape)
            self._v_b = np.zeros(gradb.shape)

        # Biased moments
        self._m_w = self._beta1 * self._m_w + (1 - self._beta1) * gradw
        self._v_w = self._beta2 * self._v_w + (1 - self._beta2) * np.power(gradw, 2)

        self._m_b = self._beta1 * self._m_b + (1 - self._beta1) * gradb
        self._v_b = self._beta2 * self._v_b + (1 - self._beta2) * np.power(gradb, 2)

        # Unbiased moments
        mhat_w = self._m_w / (1 - np.power(self._beta1, self._time_step))
        vhat_w = self._v_w / (1 - np.power(self._beta2, self._time_step))

        mhat_b = self._m_b / (1 - np.power(self._beta1, self._time_step))
        vhat_b = self._v_b / (1 - np.power(self._beta2, self._time_step))

        self._weight_updates = self._learning_rate / (np.sqrt(vhat_w) + self._eps) * mhat_w
        self._bias_updates = self._learning_rate / (np.sqrt(vhat_b) + self._eps) * mhat_b

        self._time_step += 1

        return self._weight_updates, self._bias_updates