import os
import gc
import glob
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.models import Model

def get_sequence_data(data_path):
    """Processes raw input and target texts to extract vocabs, tokens, and sequence lengths.

    Args:
        data_path: Path of data.

    Returns:
        input_texts: A dictionary of key:value pairs whose values span train and valid sets.
        target_texts: A dictionary of key:value pairs whose values span train, valid sets.
         data_character: A dictionary of configuration parameters for the Keras data generators.
    """

    # If 'file_path' is a path.
    if os.path.isdir(data_path):
        print("file_path is a directory")
        file_list = glob.glob(data_path + "*.csv")
        print("File list:", *file_list, sep="\n")
        data = []
        for file in file_list:
            df = pd.read_csv(file, header=None)
            data.append(df)
        data = pd.concat(data)
    # If 'file_path' is a file
    elif os.path.isfile(data_path):
        print("file_path is a file")
        data = pd.read_csv(data_path, header=None)
    else:
        print("Have no this file_path or file.")
        os._exit(1)

    print("#Number of datas:", len(data))
    
    raw_input = data[0]
    raw_output = data[1]
    raw_output = ['\t' + str(item) + '\n' for item in raw_output]

    input_tokens = sorted(list(set("".join(raw_input))))
    output_tokens = sorted(list(set("".join(raw_output))))
    num_input_tokens = len(input_tokens)
    num_output_tokens = len(output_tokens)
    max_input_len = max([len(expr) for expr in raw_input])
    max_output_len = max([len(expr) for expr in raw_output])
    input_token_index = dict([(char, i) for i, char in enumerate(input_tokens)])
    output_token_index = dict([(char, i) for i, char in enumerate(output_tokens)])
    print("Number of data:", len(data))
    print("Number of unique input tokens:", num_input_tokens)
    print("Number of unique target tokens:", num_output_tokens)
    print("Max sequence length of inputs:", max_input_len)
    print("Max sequence length of targets:", max_output_len)
   
    data_character = {
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": num_output_tokens,
        "max_input_len": max_input_len,
        "max_output_len": max_output_len,
        "input_token_index": input_token_index,
        "output_token_index": output_token_index,
    }

    with open("../gru_save/dataset_param.json", "w+") as f:
        json.dump(data_character, f)

    input_train, input_valid, target_train, target_valid = train_test_split(
        raw_input,
        raw_output,
        test_size=0.1,
        random_state=42,
    )

    # Free memory
    del data
    del raw_input
    del raw_output
    gc.collect()

    input_texts, target_texts = {}, {}
    input_texts["train"] = input_train
    input_texts["valid"] = input_valid
    target_texts["train"] = target_train
    target_texts["valid"] = target_valid

    return input_texts, target_texts, data_character

def match_rate(y_true, y_pred):

    # get indices from vectors
    y_pred_argmax = tf.argmax(y_pred, axis=-1)
    y_true_argmax = tf.argmax(y_true, axis=-1)

    # get mask of rows with no entry
    mask = tf.equal(tf.reduce_sum(y_true, axis=-1), 0)

    pred_match = tf.equal(y_pred_argmax, y_true_argmax)

    # if no label in y_true, then actual match doesn't matter --> equal=True
    pred_match_fixed = tf.where(
        mask, tf.ones_like(pred_match, dtype=tf.bool),
        pred_match
    )

    exact_match = tf.reduce_min(tf.cast(pred_match_fixed, tf.float32), axis=[1])
    return tf.reduce_mean(exact_match)

class Decoder(object):
    def __init__(
        self,
        model,
        hidden_dim,
        num_input_tokens,
        num_output_tokens,
        max_input_len,
        max_output_len,
        input_token_index,
        output_token_index,
        reverse_output_token_index
    ):
        self.model = model
        self.hidden_dim = hidden_dim
        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = num_output_tokens
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.input_token_index = input_token_index
        self.output_token_index = output_token_index
        self.reverse_output_token_index = reverse_output_token_index
        self.get_decoder_model()


    def get_decoder_model(self):
        encoder_inputs = self.model.input[0]   # input_1
        encoder_outputs, state_enc = self.model.layers[2].output   # lstm_1
        encoder_states = [state_enc]
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = self.model.input[1]   # input_2
        decoder_state_input_h = Input(shape=(self.hidden_dim,), name='input_3')
        # decoder_state_input_c = Input(shape=(self.hidden_dim,), name='input_4')
        decoder_states_inputs = [decoder_state_input_h]
        decoder_gru = self.model.layers[3]
        decoder_outputs, state_dec = decoder_gru(
            decoder_inputs,
            initial_state=decoder_states_inputs
        )
        decoder_states = [state_dec]
        decoder_dense = self.model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        states_value = [self.encoder_model.predict(input_seq)]

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_output_tokens))
        # Populate the first character of target sequence with the start character.
        # target_seq[0, 0, output_token_index['\t']] = 1.
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, state = self.decoder_model.predict(
                [target_seq] + states_value
            )

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_output_token_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or len(decoded_sentence) > self.max_output_len):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_output_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [state]

        return decoded_sentence

    # Define the function that encode expression to a 3 dimension matrix
    def expr_encoder(self, expr):
        matrix = np.zeros(
            (self.max_input_len, self.num_input_tokens),
            dtype='float32'
        )
        for i, char in enumerate(expr):
            matrix[i, self.input_token_index[char]] = 1.
        
        return [[matrix]]

    # Define predict Function
    def predict(self, expr):
        # expr = expr
        input_mt = self.expr_encoder(expr)
        return self.decode_sequence(input_mt)[:-1]

