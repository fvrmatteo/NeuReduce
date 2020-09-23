#!/usr/bin/env python3
# coding: utf-8

import os
import json
import pickle
import warnings
import multiprocessing
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.optimizers import Adam
from generator import Generator
from lstm import LSTM_S2S
from utils import match_rate
from utils import get_sequence_data
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

def main():

    BATCH_SIZE = 1024
    EPOCHS = 1000
    hidden_dim = 256 # latend dimension
    DATA_PATH = "../../data/linear/train/train.csv"
    # DATA_PATH = "../predataset.csv"

    cpu_count = multiprocessing.cpu_count()
    print("Number of used CPUs:", cpu_count // 2)

    input_exprs, output_exprs, data_character = get_sequence_data(DATA_PATH)

    train_generator = Generator(
        batch_size=BATCH_SIZE,
        input_exprs=input_exprs["train"],
        output_exprs=output_exprs["train"],
        **data_character
    )
    valid_generator = Generator(
        batch_size=BATCH_SIZE,
        input_exprs=input_exprs["valid"],
        output_exprs=output_exprs["valid"],
        **data_character
    )

    lstm = LSTM_S2S(
        data_character["num_input_tokens"],
        data_character["num_output_tokens"],
        hidden_dim,
    )
    model = lstm.get_model()

    adam = Adam(
        lr=0.001,
        beta_1=0.9,
        beta_2=0.995,
        epsilon=1e-9,
        decay=0.0,
        amsgrad=False,
        clipnorm=0.1,
    )

    model.compile(
        optimizer=adam,
        loss="categorical_crossentropy",
        metrics=[match_rate]
    )
    model.summary()

    # When loss doesn't change, learning will terminate after 10 epochs
    es = EarlyStopping(monitor='loss', patience=10)
    # When loss doesn't change in 5 epochs, learning rate will down 
    # to 0.2*(learning rate)
    lr = ReduceLROnPlateau(
        monitor="loss",
        factor=0.2,
        patience=5,
        verbose=1,
        mode="auto",
        # min_lr=1e-6
    )

    train_hist = model.fit_generator(
        train_generator,
        epochs=EPOCHS,
        use_multiprocessing=True,
        callbacks=[es, lr],
        validation_data=valid_generator,
        workers=cpu_count//2,
        verbose=1
    )

    # create and save plot of losses
    save_path = "../lstm_save/"

    # save model
    model.save(save_path + "model.h5")
    
    # save train history
    with open(save_path + "train_history.json", "wb") as f:
        pickle.dump(train_hist.history, f)

    plt.figure()
    plt.plot(train_hist.history["loss"], color="b", label="train")
    plt.plot(train_hist.history["val_loss"], color="r", label="valid")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc="best")
    plt.ylim([0, 1])
    plt.grid(True, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path + "losses.png", format="png")
    plt.savefig(save_path + "losses.pdf", format="pdf")

    # create and save plot of evaluation metrics
    plt.figure()
    plt.plot(train_hist.history["match_rate"], color="b", label="train")
    plt.plot(
        train_hist.history["val_match_rate"],
        color="r",
        label="valid"
    )
    plt.xlabel("epochs")
    plt.ylabel("match rate")
    plt.legend(loc="best")
    plt.ylim([0, 1])
    plt.grid(True, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path + "match_rate.png", format="png")
    plt.savefig(save_path + "match_rate.pdf", format="pdf")


if __name__ == "__main__":
    main()
