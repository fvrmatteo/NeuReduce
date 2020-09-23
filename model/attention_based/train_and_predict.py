#!/usr/bin/env python3
# coding: utf-8

import os
import glob
import pandas as pd
import numpy as np
import time
import z3
from keras.layers import Activation, dot, concatenate
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from keras.models import Model, load_model
import encoding
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras.optimizers import Adam

INPUT_LENGTH = 100
OUTPUT_LENGTH = 35 #24
HIDDEN_DIM = 256
EPOCHS = 1000
data_path = '../../data/linear/train/train.csv'
# data_path = "./predataset.csv"

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

data_input = [s for s in data[0]]
data_output = [s for s in data[1]]

data_size = len(data)

training_input  = data_input[data_size*0//100:data_size*95//100]
training_output = data_output[data_size*0//100:data_size*95//100]
validation_input = data_input[data_size*95//100:data_size*100//100]
validation_output = data_output[data_size*95//100:data_size*100//100]

print('training size', len(training_input))
print('validation size', len(validation_input))




input_encoding, input_decoding, input_dict_size = encoding.build_characters_encoding(data_input)
output_encoding, output_decoding, output_dict_size = encoding.build_characters_encoding(data_output)


encoded_training_input = encoding.transform(
    input_encoding, training_input, vector_size=INPUT_LENGTH)
encoded_training_output = encoding.transform(
    output_encoding, training_output, vector_size=OUTPUT_LENGTH)

encoded_validation_input = encoding.transform(
    input_encoding, validation_input, vector_size=INPUT_LENGTH)
encoded_validation_output = encoding.transform(
    output_encoding, validation_output, vector_size=OUTPUT_LENGTH)


encoder_input = Input(shape=(INPUT_LENGTH,))
decoder_input = Input(shape=(OUTPUT_LENGTH,))

encoder = Embedding(input_dict_size, HIDDEN_DIM, input_length=INPUT_LENGTH, mask_zero=True)(encoder_input)
encoder = LSTM(HIDDEN_DIM, return_sequences=True, unroll=True)(encoder)
encoder_last = encoder[:,-1,:]


decoder = Embedding(output_dict_size, HIDDEN_DIM, input_length=OUTPUT_LENGTH, mask_zero=True)(decoder_input)
decoder = LSTM(HIDDEN_DIM, return_sequences=True, unroll=True)(decoder, initial_state=[encoder_last, encoder_last])

attention = dot([decoder, encoder], axes=[2, 2])
attention = Activation('softmax', name='attention')(attention)
context = dot([attention, encoder], axes=[2,1])
decoder_combined_context = concatenate([context, decoder])
output = TimeDistributed(Dense(HIDDEN_DIM, activation="tanh"))(decoder_combined_context)
output = TimeDistributed(Dense(output_dict_size, activation="softmax"))(output)
    
adam = Adam(
    lr=0.001,
    beta_1=0.9,
    beta_2=0.995,
    epsilon=1e-9,
    decay=0.0,
    amsgrad=False,
    clipnorm=0.1,
)

lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=5,
    verbose=1,
    mode="auto",
    # min_lr=1e-6
)

model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
model.compile(optimizer=adam, loss='categorical_crossentropy')

model.summary()

# from keras.utils import plot_model
# plot_model(model, to_file='model.png',show_shapes=True)
# exit()

training_encoder_input = encoded_training_input
training_decoder_input = np.zeros_like(encoded_training_output)
training_decoder_input[:, 1:] = encoded_training_output[:,:-1]
training_decoder_input[:, 0] = encoding.CHAR_CODE_START
training_decoder_output = np.eye(output_dict_size)[encoded_training_output.astype('int')]

validation_encoder_input = encoded_validation_input
validation_decoder_input = np.zeros_like(encoded_validation_output)
validation_decoder_input[:, 1:] = encoded_validation_output[:,:-1]
validation_decoder_input[:, 0] = encoding.CHAR_CODE_START
validation_decoder_output = np.eye(output_dict_size)[encoded_validation_output.astype('int')]


es = EarlyStopping(monitor='val_loss', patience=10)

if os.path.isfile('model.h5'):
    model = load_model('model.h5')
else:
    train_hist = model.fit(x=[training_encoder_input, training_decoder_input], y=[training_decoder_output],
          validation_data=([validation_encoder_input, validation_decoder_input], [validation_decoder_output]),
          verbose=1, batch_size=1024, epochs=EPOCHS,
          callbacks=[es, lr])

# model.fit(x=[training_encoder_input, training_decoder_input], y=[training_decoder_output],
#           validation_data=([validation_encoder_input, validation_decoder_input], [validation_decoder_output]),
#           verbose=1, batch_size=1024, epochs=200,
#           callbacks=[es])

# model.save('model.h5')

# with open("train_history.json", "wb") as f:
#     pickle.dump(train_hist.history, f)

# plt.figure()
# plt.plot(train_hist.history["loss"], color="b", label="train")
# plt.plot(train_hist.history["val_loss"], color="r", label="valid")
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.legend(loc="best")
# plt.ylim([0, 1])
# plt.grid(True, linestyle="--")
# plt.tight_layout()
# plt.savefig(save_path + "losses.png", format="png")
# plt.savefig(save_path + "losses.pdf", format="pdf")


def generate(text):
    encoder_input = encoding.transform(input_encoding, [text.lower()], INPUT_LENGTH)
    decoder_input = np.zeros(shape=(len(encoder_input), OUTPUT_LENGTH))
    decoder_input[:,0] = encoding.CHAR_CODE_START
    for i in range(1, OUTPUT_LENGTH):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:,i] = output[:,i]
    return decoder_input[:,1:]

def decode(decoding, sequence):
    text = ''
    for i in sequence:
        if i == 0:
            break
        text += output_decoding[i]
    return text

def predict(text):
    decoder_output = generate(text)
    return decode(output_decoding, decoder_output[0])

def verify_equivalent(source_expr, targ_expr, bitnumber=8):
    x, y, z, a, b, c, d, e = z3.BitVecs("x y z a b c d e", bitnumber)
    try:
        leftEval = eval(source_expr)
        rightEval = eval(targ_expr)
    except:
        return "unsat"
    solver = z3.Solver()
    solver.add(leftEval != rightEval)
    result = solver.check()

    return str(result)

print(predict('(x&y)-(~x&y)+(x^y)+3*(~(x|y))-(~(x^y))-(x|~y)-(~x)-1'))
exit()

f = open("../attention_lstm_save/predict_result.txt", "w+")
path = "../../data/linear/test/test_data.csv"
test_data = pd.read_csv(path, header=None)
mba_exprs, targ_exprs = test_data[0], test_data[1]

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
    predict_expr = predict(mba_exprs[idx])
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
print("Average result length: %.4f" % (total_len/test_count))
    
f.close()
