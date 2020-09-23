#!/usr/bin/env python3
# coding: utf-8

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from z3 import *
from keras.models import load_model
from pickle import load
from utils import get_sequence_data, CharacterTable


def main():
    REVERSE = True

    file_path = "../predataset.csv"
    # file_path = "../../data/linear/test/test_data.csv"
    ds = pd.read_csv(file_path, header=None)
    input_ds = ds[0]
    output_ds = ds[1]
    
    with open("../rnn_save/dataset_param.json", "r") as file:
        setting = json.load(file)
    input_tokens = setting["input_tokens"]
    max_input_len = setting["max_input_len"]

    ctable = CharacterTable(input_tokens)

    # Load model from h5 file
    model = load_model("../rnn_save/model.h5")

    model.summary()

    correct_predict_count = 0
    total_test_count = 1000
    time_sum = 0
    max_time = 0
    min_time = 1
    for i in range(total_test_count):
        mba_expr = input_ds[i]
        simp_expr = output_ds[i]
        start_time = time.time()
        if REVERSE:
            x = np.flipud(ctable.encode(mba_expr, max_input_len, reverse=True))
        else:
            x = ctable.encode(mba_expr, max_input_len)
        
        y_pred = model.predict_classes([[x]], verbose=0)
        predict = ctable.decode(y_pred[0], calc_argmax=False)
        end_time = time.time()
        consume_time = end_time - start_time
        time_sum += consume_time
        if max_time < consume_time:
            max_time = consume_time
        if min_time > consume_time:
            min_time = consume_time
        # print('=' * 50)
        # print('M', mba_expr)
        # print('T', simp_expr)
        # print('P', predict, end=' ')
        print("No.%d" % (i+1), end=' ')
        if simp_expr == predict.strip():
            print("\033[1;32m O \033[0m", end=' ')
            correct_predict_count += 1
        else:
            print("\033[1;31m X \033[0m", end=' ')
        print("Time = %.4f" % consume_time)

        # rowx, rowy = x[np.array([ind])], y[np.array([ind])]
    #     preds = model.predict_classes(rowx, verbose=0)
    #     mba_expr = ctable.decode(rowx[0])
    #     targ_expr = ctable.decode(rowy[0])
    #     predict = ctable.decode(preds[0], calc_argmax=False)
    #     print('M', mba_expr[::-1] if REVERSE else mba_expr)
    #     print('T', targ_expr)
    #     print('P', predict, end=' ')
    #     if targ_expr == predict:
    #         print(colors.ok + 'O' + colors.close)
    #         correct_predict_count += 1
    #     else:
    #         print(colors.fail + 'X' + colors.close)
    # print("C/T: %d/%d" % (correct_predict_count, total_test_count))
    print("#Correct predict: %d/%d" % (correct_predict_count, total_test_count))
    print("Correct rate: %.4f" % (correct_predict_count/total_test_count))
    print("Average solve time: %.4f" % (time_sum/total_test_count))
    print("Maximum solve time: %.4f" % (max_time))
    print("Minimum solve time: %.4f" % (min_time))
        
if __name__ == "__main__":
    main()