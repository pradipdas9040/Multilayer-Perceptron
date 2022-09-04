from common import np
import common


# loss functions calculate loss for each datapoint
# to get the mean loss over all data points call the mean method


class Loss(object):
    def __init__(self): pass

    @staticmethod
    def loss(y1: np.ndarray, y2: np.ndarray) -> float:
        # calculate loss
        # returns a column vector of losses for each datapoint
        pass

    @staticmethod
    def gradient(y1: np.ndarray, y2: np.ndarray) -> float:
        # return the gradients for each datapoint
        # as a matrix of row vectors (gradients)
        # OR
        # return the jacobian for each datapoint
        # as a 3D matrix of 2D matrices (jacobians)
        pass

    @classmethod
    def mean_loss(cls, y1: np.ndarray, y2: np.ndarray) -> float:
        """
        mean_loss = 1/n * sum(error_vector)
        n = no of data points
        :param y1: (Output) Matrix of row vectors containing output data points
        :param y2: (Desired output) Matrix of row vectors containing desired output data points
        :return: mean loss
        """
        a = np.sum(cls.loss(y1=y1, y2=y2)) / len(y1)
        if common.cupy:
            # cupy returns an array instead of
            a = a.tolist()

        return a


class SquaredError(Loss):
    __name__ = 'squarederror'

    @staticmethod
    def loss(y1: np.ndarray, y2: np.ndarray) -> float:
        """
        mean square error
        1/2 * || y1 - y2 ||^2
        :param y1: (Predicted Output) Matrix of row vectors containing output data points
        :param y2: (Desired output) Matrix of row vectors containing desired output data points
        :return: squared errors for each datapoint as a column vector
        """
        return 0.5 * np.square(np.linalg.norm(y1 - y2, axis=1, keepdims=True))

    @staticmethod
    def gradient(y1: np.ndarray, y2: np.ndarray) -> float:
        """
        gradient = predicted_output_vector - desired_output_vector
        :param y1: (Predicted Output) Matrix of row vectors containing output data points
        :param y2: (Desired output) Matrix of row vectors containing desired output data points
        :return: squared error loss gradients as a column vector
        """
        return y1 - y2


class CrossEntropy(Loss):
    __name__ = 'cross-entropy'
    @staticmethod
    def loss(y1: np.ndarray, y2: np.ndarray) -> float:
        """
        Cross entropy loss (for one datapoint)
        - sum(y2_i * log(y1_i))
        :param y1: (Predicted output) Matrix of row vectors containing output data points
                   (values must be b/w 0 and 1 and sum to 1)
        :param y2: (Desired output) Matrix of row vectors containing desired output data points
                   (values must be b/w 0 and 1 and sum to 1)
        :return: cross entropy losses as a column vector
        """

        # TODO: Add check for values b/w 0 and 1 and sum to 1
        # TODO: log function raises divide by zero error when one of the values in y1 is 0
        return -np.sum(y2 * np.log2(y1), axis=1, keepdims=True)

    @staticmethod
    def gradient(y1: np.ndarray, y2: np.ndarray) -> float:
        """
        gradient = desired_output_vector / predicted_output_vector
        :param y1: (Predicted output) Matrix of row vectors containing output data points
                   (values must be b/w 0 and 1 and sum to 1)
        :param y2: (Desired output) Matrix of row vectors containing desired output data points
                   (values must be b/w 0 and 1 and sum to 1)
        :return: cross entropy loss gradients as a column vector
        """

        return - y2 / y1