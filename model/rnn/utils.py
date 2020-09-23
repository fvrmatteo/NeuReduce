import os
import gc
import glob
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.

        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows, reverse=False):
        """One-hot encode given string C.

        Because the token index of space is always 0, so we make column 0 to be all 1

        # Arguments
            C: string, to be encoded.
            num_rows: Number of rows in the returned one-hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        ones = np.ones(num_rows)
        x[:, 0] = ones
        for i, c in enumerate(C[::-1] if reverse else C):
            x[i, self.char_indices[c]] = 1
            x[i, 0] = 0
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.

        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of character indices (used with `calc_argmax=False`).
            calc_argmax: Whether to find the character index with maximum
                probability, defaults to `True`.
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

def get_sequence_data(file_path):
    # If 'file_path' is a path.
    if os.path.isdir(file_path):
        print("file_path is a direction")
        file_list = glob.glob(file_path + "*.csv")
        print("File list:", *file_list, sep="\n")
        ds = []
        for file in file_list:
            df = pd.read_csv(file, header=None)
            ds.append(df)
        ds = pd.concat(ds)
    # If 'file_path' is a file
    elif os.path.isfile(file_path):
        print("file_path is a file")
        ds = pd.read_csv(file_path, header=None)
    else:
        print("Have no this file_path or file.")
        os._exit(1)

    print("#Number of datas:", len(ds))

    input_ds = ds[1]
    output_ds = ds[0]

    del ds
    gc.collect()

    # Maximum length of mba expression and target expression
    max_input_len = max([len(expr) for expr in input_ds])
    max_output_len = max([len(expr) for expr in output_ds])
    input_tokens = sorted(list(set(' '.join(input_ds))))
    num_input_tokens = len(input_tokens)

    data_character = {
        "max_input_len": max_input_len,
        "max_output_len": max_output_len,
        "input_tokens": input_tokens,
        "num_input_tokens": num_input_tokens
    }
    with open("../rnn_save/dataset_param.json", "w+") as file:
        json.dump(data_character, file)

    input_train, input_valid, output_train, output_valid = train_test_split(
        input_ds,
        output_ds,
        test_size=0.2,
        random_state=42,
    )

    input_exprs, output_exprs = {}, {}
    input_exprs['train'] = input_train
    input_exprs['valid'] = input_valid
    output_exprs['train'] = output_train
    output_exprs['valid'] = output_valid

    return input_exprs, output_exprs, data_character
