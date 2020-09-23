import keras
import numpy as np
from utils import CharacterTable

class Generator(keras.utils.Sequence):
    def __init__(
        self,
        batch_size,
        input_exprs,
        output_exprs,
        max_input_len,
        max_output_len,
        input_tokens,
        num_input_tokens,
        reverse=True,
        shuffle=True,
    ):

        self.batch_size = batch_size
        self.input_exprs = input_exprs
        self.output_exprs = output_exprs
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.input_tokens = input_tokens
        self.num_input_tokens = num_input_tokens
        self.indexes = list(range(len(self.input_exprs)))
        self.reverse = reverse
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.input_exprs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        return self.__data_generation(indexes)

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.input_exprs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        "Generates data containing batch_size samples"
        ctable = CharacterTable(self.input_tokens)
        x = np.zeros(
            (
                self.batch_size, 
                self.max_input_len, 
                self.num_input_tokens
            ),
            dtype="float32"
        )
        y = np.zeros(
            (
                self.batch_size,
                self.max_output_len,
                self.num_input_tokens
            ),
            dtype="float32"
        )
        input_exprs = list(self.input_exprs)
        output_exprs = list(self.output_exprs)
        batch_inputs = [input_exprs[i] for i in indexes]
        batch_outputs = [output_exprs[i] for i in indexes]

        for i, expr in enumerate(batch_inputs):
            if self.reverse:
                x[i] = np.flipud(ctable.encode(expr, self.max_input_len, reverse=True))
            else:
                x[i] = ctable.encode(expr, self.max_input_len)
        for i, expr in enumerate(batch_outputs):
            y[i] = ctable.encode(expr, self.max_output_len)

        return (x, y)
