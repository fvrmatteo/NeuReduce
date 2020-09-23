import keras
import numpy as np

class Generator(keras.utils.Sequence):
    def __init__(
        self,
        batch_size,
        input_exprs,
        output_exprs,
        max_input_len,
        max_output_len,
        num_input_tokens,
        num_output_tokens,
        input_token_index,
        output_token_index,
        shuffle=True,
    ):

        self.batch_size = batch_size
        self.input_exprs = input_exprs
        self.output_exprs = output_exprs
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = num_output_tokens
        self.input_token_index = input_token_index
        self.output_token_index = output_token_index
        self.indexes = list(range(len(self.input_exprs)))
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
        encoder_input_data = np.zeros(
            (
                self.batch_size,
                self.max_input_len,
                self.num_input_tokens
            ),
            dtype="float32",
        )
        decoder_input_data = np.zeros(
            (
                self.batch_size,
                self.max_output_len,
                self.num_output_tokens,
            ),
            dtype="float32",
        )
        decoder_target_data = np.zeros(
            (
                self.batch_size,
                self.max_output_len,
                self.num_output_tokens,
            ),
            dtype="float32",
        )

        input_exprs = list(self.input_exprs)
        output_exprs = list(self.output_exprs)
        batch_inputs = [input_exprs[i] for i in indexes]
        batch_outputs = [output_exprs[i] for i in indexes]

        for i, (input_text, target_text) in enumerate(zip(batch_inputs, batch_outputs)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, self.input_token_index[char]] = 1.0
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, self.output_token_index[char]] = 1.0
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, self.output_token_index[char], ] = 1.0

        return ([encoder_input_data, decoder_input_data], decoder_target_data)
