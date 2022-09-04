import sys
from typing import List, Tuple
from common import np
import sklearn
from sklearn.model_selection import train_test_split

from constants import Defaults, Losses as loss_fn_names
from functions import loss as loss_functions
from layers import Dense
import common


class Model:
    def __init__(self):
        self._input_size = 0
        self.n_classes = 0  # output size
        self._n_layers = 0  # number of layers (hidden layers + 1 for output)

        self.layers: List[Dense] = []
        # self.add_layer.__doc__ = Layer.__doc__
        self._loss_fn: loss_functions.Loss = None

    def add_layer(self, input_size, output_size, act_fn, weights=None, biases=None):
        """
        Add a layer to the network
        :param input_size: Input size of layer
        :param output_size: Output size of layer
        :param act_fn: Activation function of layer
                'identity', 'sigmoid', 'relu', 'tanh', 'softmax'
        :param weights: Weights (input size x output size)
        :param biases: Biases (1 X output size)
        """
        layer = Dense(input_size=input_size,
                      n_nodes=output_size,
                      act_fn=act_fn,
                      weights=weights,
                      biases=biases)

        if len(self.layers) == 0:
            self._input_size = input_size

        self.n_classes = output_size

        self.layers.append(layer)

    def __set_layer_optimizers(self, optimizer, learning_rate=Defaults.learning_rate,
                               momentum_factor=Defaults.momentum_factor, beta1=Defaults.beta1, beta2=Defaults.beta2):
        for layer in self.layers:
            layer.set_optimizer(optimizer=optimizer, learning_rate=learning_rate, momentum_factor=momentum_factor,
                                beta1=beta1, beta2=beta2)

    def fit(self, x: np.ndarray, y: np.ndarray, *, validate_x=None, validate_y=None, validate_frac=0.1,
            epochs=Defaults.epochs,
            loss=Defaults.loss, learning_rate=Defaults.learning_rate, batch_size=Defaults.batch_size,
            momentum_factor=Defaults.momentum_factor, beta1=Defaults.beta1, beta2=Defaults.beta2,
            optimizer=Defaults.optimizer, threshold=1e-5):
        """
        Fit an input and output
        :param x: input matrix
                [[----- datapoint1 --------],
                 [----- datapoint2 --------],
                            .
                            .
                            .
                [------ datapointk --------]]
        :param y: output vector
                [[------ y1 ------],
                 [------ y2 ------],
                          .
                          .
                          .
                [------- yk ------]]
        :param validate_x: Validate set x
                If not provided, validate set is constructed wrt validate_frac
        :param validate_y: Validate set y
                If not provided, validate set is constructed wrt validate_frac
        :param validate_frac: Fraction of x and y to set aside for validation
        :param epochs: Epochs
        :param loss: Loss function to use
                'squarederror': squared error
                'softmax': Softmax
        :param learning_rate: Learning rate
        :param batch_size: Batch size to use for training
        :param momentum_factor: Momentum factor
        :param beta1: beta1 for Adam
        :param beta2: beta2 for Adam
        :param optimizer: Optimizer to use
                'gd': Gradient Descent
                'mgd': Momentum based gradient descent
                'rmsprop': RMSProp
                'adagrad': Adagrad
                'adam': Adam
        :param threshold: Maximum allowed loss
        :return: Validation loss and accuracy history
        """
        self.__set_layer_optimizers(optimizer=optimizer, learning_rate=learning_rate, momentum_factor=momentum_factor,
                                    beta1=beta1, beta2=beta2)
        self._loss_fn = self.__select_loss_function(loss)

        validate = False

        if validate_x is not None:
            validate = True
        elif validate_frac != 0:
            x, validate_x, y, validate_y = train_test_split(x, y, test_size=validate_frac)
            validate = True

        # convert everything to cupy array
        x = np.array(x)
        y = np.array(y)

        if validate:
            validate_x = np.array(validate_x)
            validate_y = np.array(validate_y)

        loss_stash = []  # stash for loss
        acc_stash = []  # stash for accuracy

        min_loss_seen = 1e8
        tol = 0

        for epoch in range(epochs):
            sklearn.utils.shuffle(x, y)

            # for each batch
            for i in range(int(np.ceil(len(x) / batch_size))):
                start_index = batch_size * i
                end_index = min(batch_size * (i + 1), len(x)) - 1
                # ceil because no of datapoints might not be an exact multiple of batch size
                # min because no of datapoints available in the last batch might be less

                input_, output = x[start_index:end_index + 1], y[start_index:end_index + 1]

                self.forward_pass(input_)
                self.backward_pass(output)

                # update weight
                self.__update_weights()

            self._log('epoch ', epoch, end='')
            if validate:
                loss, acc = self.__measure_metrics(validate_x, validate_y)
                loss_stash.append(loss)
                acc_stash.append(acc)
                self._log(' loss = ', loss, 'acc = ', acc, end='')

            self._log()

            if validate and loss <= threshold:
                self._log('Reached the desired loss. Stopping training.')
                break

            if validate:
                if loss <= min_loss_seen:
                    tol = 0
                    min_loss_seen = loss
                else:
                    tol += 1
                    if tol > 5:
                        break

        return loss_stash, acc_stash

    def forward_pass(self, input_):
        """
        Performs backward pass for a batch
        :param input_: Input batch
        :return: Output batch
        """
        for layer in self.layers:
            input_ = layer.forward_pass(input_)

        return self.layers[-1].output

    def backward_pass(self, desired_output) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Performs backward pass for a batch
        :param desired_output: Desired output batch
        :return: weight_updates , bias_updates
        """

        grad = self._loss_fn.gradient(self.layers[-1].output, desired_output)

        weight_updates = []
        bias_updates = []

        for layer in self.layers[::-1]:
            layer.backward_pass(grad)
            grad = layer.downstream_grad

            weight_updates = [layer.grad_weight] + weight_updates
            bias_updates = [layer.grad_bias] + bias_updates

        return weight_updates, bias_updates

    def __update_weights(self):
        for layer in self.layers:
            layer.update_weights_biases()

    @staticmethod
    def __select_loss_function(loss: str) -> loss_functions.Loss:
        if loss == loss_fn_names.squarederror:
            return loss_functions.SquaredError()
        if loss == loss_fn_names.crossentropy:
            return loss_functions.CrossEntropy()

        raise TypeError('Unknown loss function')

    def _predict(self, x) -> np.ndarray:
        output = self.forward_pass(np.array(x))
        return output

    def predict(self, x):
        return np.array(self._predict(x))

    def __measure_metrics(self, x: np.ndarray, y: np.ndarray):
        # returns loss and accuracy
        y_pred = self._predict(x)

        return self._loss_fn.mean_loss(y_pred, y), \
               common.accuracy(common.convert_to_binary(y), common.convert_to_binary(y_pred))

    def _log(self, *text, file=sys.stdout, **kwargs):
        print(*text, file=file, **kwargs)

    def dump_weights(self):
        pass

    def dump_model(self):
        pass

    def load_weights(self):
        pass

    def load_model(self):
        pass