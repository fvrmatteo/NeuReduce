from keras.layers import (
    Input,
    Dense,
    SimpleRNN,
    SimpleRNNCell,
    Embedding,
    Dropout,
    RepeatVector,
    TimeDistributed
)
from keras.models import Model, Sequential
import tensorflow as tf

class RNN_S2S:
    def __init__(
        self, 
        units,
        num_input_tokens, 
        max_input_len,
        max_output_len,
        rnn_layers=4
    ):
        self.units = units
        self.num_input_tokens = num_input_tokens
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.rnn_layers = rnn_layers

    def get_model(self):
        model = Sequential()
        # "Encode" the input sequence using an RNN, producing an output of self.units.
        model.add(SimpleRNN(self.units, input_shape=(self.max_input_len, self.num_input_tokens)))
        # As the decoder RNN's input, repeatedly provide with the last output of
        # RNN for each time step. Repeat max_output_len times as that's the maximum
        # length of output.
        model.add(RepeatVector(self.max_output_len))
        # The decoder RNN could be multiple layers stacked or a single layer.
        for _ in range(self.rnn_layers):
            # By setting return_sequences to True, return not only the last output but
            # all the outputs so far in the form of (num_samples, timesteps,
            # output_dim). This is necessary as TimeDistributed in the below expects
            # the first dimension to be the timesteps.
            model.add(SimpleRNN(self.units, return_sequences=True))

        model.add(SimpleRNN(self.units, return_sequences=True))
        # Apply a dense layer to the every temporal slice of an input. For each of step
        # of the output sequence, decide which character should be chosen.
        model.add(TimeDistributed(Dense(self.num_input_tokens, activation='softmax')))
        return model
