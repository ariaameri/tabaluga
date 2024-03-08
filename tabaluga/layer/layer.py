from keras.layers import Layer


class Layer(Layer):
    """The base/abstract class for custom defined neural layers."""

    def __init__(self):
        """Initializes the custom neural layer."""

        pass

    def build(self, input_shape):
        """Builds the layer.

        This method knows the shape of the input and does the rest of
        the initializations such as the layer weights.

        Parameters
        ----------
        input_shape
            Keras tensor (future input to layer)
            or list/tuple of Keras tensors to reference
            for weight shape computations.
        """

    def call(self, inputs, **kwargs):
        """The feedforward of the layer.

        Parameters
        ----------
        inputs
            Input tensor, or list/tuple of input tensors.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        A tensor or list/tuple of tensors.
        """