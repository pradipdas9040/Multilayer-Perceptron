from typing import Union, List
from common import np
from constants import Defaults, Activations as act_fn_names, Optimizers as opt_names
from functions import activations
from optimizer import adam as opt


class Dense:
    __name__ = 'Dense'

    def __init__(self, input_size: int, n_nodes: int, act_fn: str,
                 weights: np.ndarray = None, biases: Union[List[float], np.ndarray] = None):
        """
        Layer class
        :param input_size: Size of input vector to the layer
        :param n_nodes: No of nodes (also the size of output array)
        :param act_fn: Activation function
                'tanh' , 'sigmoid', 'relu', 'softmax', 'identity'
        :param weights: Weight array to initialize
        :param biases: Biases as list or array to initialize
        :param optimizer: Optimizer to use
                'gd', 'mgd', 'nag', 'rmsprop', 'adagrad', 'adam'
        :param learning_rate: Learning rate
        :param momentum_factor: Momentum Factor
        :param beta1: 1st moment decay rate (for adam)
        :param beta2: 2nt moment decay rate (for adam)
        """

        self._input_size = input_size
        self._n_nodes = self._output_size = n_nodes
        self._act_fn_name = act_fn
        self._act_fn = self.__select_act_fn(act_fn)
        self._weights: np.ndarray = None  # weight array
        self._biases: np.ndarray = None  # bias row vector
        self.__initialise_weights(weights=weights)
        self.__initialise_biases(biases=biases)
        self.__optimizer = opt.Adam()

        # stash
        self.input: np.ndarray = None  # stash for input row vectors
        self.preactivated_output: np.ndarray = None  # stash for pre activated output row vectors
        self.output: np.ndarray = None  # stash for output row vectors

        self.upstream_grad: np.ndarray = None  # stash for upstream grad row vectors (dL/dy) (received from next layer)
        self.grad_weight: np.ndarray = None  # stash for most recent weight update matrix (dL/dw)
        # saves the weight updates as received from backprop. These must be subtracted from the current weights
        # (possibly with a learning rate)
        self.weight_updates_history: np.ndarray = None  # stash for previous weight update matrix
        # saves the amount of weight updates made when weights were updated last time
        # (these were subtracted from the weights)
        self.grad_bias: np.ndarray = None  # stash for most recent bias update row vector (dL/db)
        self.bias_updates_history: np.ndarray = None  # stash for previous bias update row vector
        self.downstream_grad: np.array = None  # stash for downstream grad row vectors (dL/dx)

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    def set_optimizer(self, optimizer, learning_rate=Defaults.learning_rate, momentum_factor=Defaults.momentum_factor,
                      beta1=Defaults.beta1, beta2=Defaults.beta2):
        # Changes the optimizer
        # Use with caution. If the optimizer is changed mid-learning.
        self.__optimizer = self.__select_optimizer(optimizer, learning_rate=learning_rate,
                                                   momentum_factor=momentum_factor, beta1=beta1, beta2=beta2)

    @staticmethod
    def __select_act_fn(fn_name: str):
        # TODO: Have an option to take in vectorised function as input so that it can be shared across layers
        if fn_name == act_fn_names.tanh:
            return activations.Tanh

        if fn_name == act_fn_names.sigmoid:
            return activations.Sigmoid

        if fn_name == act_fn_names.relu:
            return activations.ReLu

        if fn_name == act_fn_names.softmax:
            return activations.Softmax

        if fn_name == act_fn_names.identity:
            return activations.Identity

        raise TypeError(f'Unknown activation function \'{fn_name}\'')

    @staticmethod
    def __select_optimizer(optimizer, learning_rate=Defaults.learning_rate, momentum_factor=Defaults.momentum_factor,
                           beta1=Defaults.beta1, beta2=Defaults.beta2):
        if optimizer == opt_names.gd:
            return opt.GradientDescent(learning_rate=learning_rate)
        if optimizer == opt_names.mgd:
            return opt.MomentumGradientDescent(learning_rate=learning_rate, momentum_factor=momentum_factor)
        if optimizer == opt_names.nag:
            return opt.NAG(learning_rate=learning_rate)
        if optimizer == opt_names.rmsprop:
            return opt.RMSProp(learning_rate=learning_rate)
        if optimizer == opt_names.adagrad:
            return opt.AdaGrad(learning_rate=learning_rate)
        if optimizer == opt_names.adam:
            return opt.Adam(beta1=beta1, beta2=beta2, learning_rate=learning_rate)

        raise TypeError(f'Unknown optimizer function \'{optimizer}\'')

    def reset_weights(self):
        self.__initialise_weights()
        self.__initialise_biases()

    def __initialise_weights(self, weights: np.ndarray = None):
        # initialise weights. If weights is not None use that value
        if weights is not None:
            self._weights = weights
            return

        self._weights = np.random.randn(self._input_size, self._output_size) * np.sqrt(2 / self._input_size)

    def __initialise_biases(self, biases: Union[List[float], np.ndarray] = None):
        # initialise biases. If biases is not None use that value
        # [[b1 b2 ---- b_n_nodes]]
        # bias is a row vector
        if biases is not None:
            self._biases = np.array([biases])
            return

        self._biases = np.zeros((1, self._n_nodes))

    def load_weights(self, weights: np.ndarray, biases: Union[List[float], np.ndarray]):
        # TODO: check sizes of weights and biases
        self._weights = weights
        self._biases = biases

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for multiple input vectors
        n = no of data points
        :param x: input vector of size (n, input_size)
        :return: output array of size (n, n_nodes)
        """
        # bias addition is broadcasted over all inputs

        self.input = x
        self.preactivated_output = x @ self._weights + self._biases
        self.output = self._act_fn.apply(self.preactivated_output)

        return self.output

    def backward_pass(self, gradients: np.ndarray):
        """
        Backward pass for a batch of input vectors for which forward pass has already been done
        n = no of data points
        :param gradients: Upstream gradients vector of size (n, n_nodes)  (dL/dy)
        :return:
        """
        # x = input vectors (matrix)
        # w = weight matrix
        # b = bias matrix
        # y = output vectors (matrix)

        self.upstream_grad = gradients  # dL/dy

        # derivative wrt pre activated output
        if self._act_fn_name == act_fn_names.softmax:
            # derivative = upstream_grad @ Jacobian.T (for each datapoint)
            # add axis to make grad 3d from 2d
            # swap 2nd and 3rd axes of jacobian to take transpose for each jacobian
            d = self.upstream_grad[:, np.newaxis, :] @ \
                self._act_fn.jacobian(self.preactivated_output, self.output).swapaxes(1, 2)
            # convert 3D matrix back to 2D
            d = d[:, 0, :]

            # jacobian for softmax is symmetric, so no need of transpose but it has been retained to have generalisation

        elif self._act_fn_name in (act_fn_names.tanh, act_fn_names.sigmoid, act_fn_names.relu, act_fn_names.identity):
            # derivative = gradient_vector * upstream_grad (for each datapoint)
            d = self._act_fn.gradient(self.preactivated_output, self.output) * self.upstream_grad
        else:
            raise TypeError

        # derivative wrt weights = a matrix  (dL/dW)
        self.grad_weight = self.input.T @ d

        # derivative wrt biases = a vector  (dL/db)
        self.grad_bias = np.sum(d, axis=0, keepdims=True)

        # derivative wrt inputs = a vector (dL/dx) (passed to the previous layer as upstream grads)
        self.downstream_grad = d @ self._weights.T

    def update_weights_biases(self):
        """
        Update weights and biases
        :return:
        """
        weight_updates, bias_updates = \
            self.__optimizer.calculate_update(gradw=self.grad_weight, gradb=self.grad_bias)

        self.weight_updates_history = weight_updates
        self.bias_updates_history = bias_updates

        self._weights -= weight_updates
        self._biases -= bias_updates