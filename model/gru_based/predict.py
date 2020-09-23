#!/usr/bin/env python
# coding: utf-8

import os
import z3
import json
import time
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from utils import Decoder
from utils import match_rate
from keras.models import load_model
from pickle import load

def verify_equivalent(source_expr, targ_expr, bitnumber=16):
    # x, y, z, a, b, c, d, e = z3.BitVecs("x y z a b c d e", bitnumber)
    x,y = z3.BitVecs('x y', bitnumber)
    try:
        leftEval = eval(source_expr)
        rightEval = eval(targ_expr)
    except:
        return "unsat"
    solver = z3.Solver()
    solver.add(leftEval != rightEval)
    result = solver.check()

    return str(result)

def main():
    # Load setting from json file
    with open('../gru_save/dataset_param.json', 'r') as f:
        setting = json.load(f)
    
    f = open("../gru_save/result.txt", "w+")
        
    num_input_tokens = setting['num_input_tokens']
    num_output_tokens = setting['num_output_tokens']
    max_input_len = setting['max_input_len']
    max_output_len = setting['max_output_len']
    input_token_index = setting['input_token_index']
    output_token_index = setting['output_token_index']
    # hidden_dim = setting['hidden_dim']
    hidden_dim = 256

    reverse_input_token_index = dict((i, char) for char, i in input_token_index.items())
    reverse_output_token_index = dict((i, char) for char, i in output_token_index.items())

    # Load model from h5 file
    model = load_model(
        "../gru_save/model.h5",
        custom_objects={"match_rate":match_rate}
    )

    # model.summary()

    decoder = Decoder(
        model=model,
        hidden_dim=hidden_dim,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        input_token_index=input_token_index,
        output_token_index=output_token_index,
        reverse_output_token_index=reverse_output_token_index
    )

    # Test1
    # mba_expr = "(x^y)-((~x&y)+(~x&y))"
    # result = decoder.predict(mba_expr)
    # print(result)
    # result2 = decoder.predict(result)
    # # print(verify_equivalent('(x^y)', mba_expr))
    # # print(verify_equivalent("(x^y)", result))
    # # print(verify_equivalent(mba_expr, result))
    # exit()
    # # Test2
    path = "../../data/linear/test/test_data.csv"
    test_data = pd.read_csv(path, header=None)
    mba_exprs, targ_exprs = test_data[0], test_data[1]

    wrong_predict_statistic = []
    correct_predict_count = 0
    z3_verify_correct_count = 0
    test_count = len(test_data)
    time_sum = 0
    max_time = 0
    min_time = 1
    total_len = 0
    for idx in range(test_count):
        print("No.%d" % (idx + 1), end=' ', file=f)
        print("No.%d" % (idx + 1), end=' ')
        print("=" * 50, file=f)
        print("MBA expr:", mba_exprs[idx], file=f)
        print("Targ expr:", targ_exprs[idx], file=f)
        start_time = time.time()
        predict_expr = decoder.predict(mba_exprs[idx])
        total_len += len(predict_expr)
        print("Pred expr:", predict_expr, file=f)
        end_time = time.time()
        consume_time = end_time - start_time
        time_sum += consume_time
        if max_time < consume_time:
            max_time = consume_time
        if min_time > consume_time:
            min_time = consume_time
        if predict_expr == targ_exprs[idx]:
            print("Predict \033[1;32m True \033[0m")
            print("Predict True", file=f)
            correct_predict_count += 1
        else:
            z3Result = verify_equivalent(predict_expr, targ_exprs[idx])
            if z3Result != 'unsat':
                print("Predict \033[1;31m False \033[0m")
                print("Predict False", file=f)
                wrong_predict_statistic.append([mba_exprs[idx], targ_exprs[idx], predict_expr])
            else:
                z3_verify_correct_count += 1
                print("Predict \033[1;33m Z3 True \033[0m")
                print("Predict Z3 True", file=f)
        print("Time = %.4f" % consume_time, file=f)
        print("", file=f)
    print("#Correct predict: %d/%d" % (correct_predict_count, test_count), file=f)
    print("#False predict true Z3:", z3_verify_correct_count, file=f)
    print("#Correct rate: %.4f" % ((correct_predict_count+z3_verify_correct_count)/test_count), file=f)
    print("Average solve time: %.4f" % (time_sum / test_count), file=f)
    print("Maximum solve time: %.4f" % (max_time), file=f)
    print("Minimum solve time: %.4f" % (min_time), file=f)
    print("Average result length: %.4f" % (total_len/test_count), file=f)
    pd.DataFrame(wrong_predict_statistic).to_csv("../gru_save/wrong_predict_statistic.csv", mode='w+', header=False, index=False)

    f.close()

if __name__ == "__main__":
    main()
