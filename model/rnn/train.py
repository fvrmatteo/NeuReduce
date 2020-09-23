#!/usr/bin/env python3

import os
import pickle
import multiprocessing
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from rnn import RNN_S2S
from generator import Generator
from utils import get_sequence_data
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from keras.callbacks import TensorBoard

def main():
    cpu_count = multiprocessing.cpu_count()
    print("Number of used CPUs:", cpu_count // 2)
    REVERSE = True
    BATCH_SIZE = 1024
    EPOCHS = 1000
    ITERRATION = 1
    RNN_LAYERS = 1
    latent_dim = 256
    file_path = "../../data/linear/train/train.csv"
    # file_path = "../predataset.csv"

    input_exprs, output_exprs, data_character = get_sequence_data(file_path)

    max_input_len = data_character["max_input_len"]
    max_output_len = data_character["max_output_len"]
    input_tokens = data_character['input_tokens']
    num_input_tokens = data_character['num_input_tokens']

    train_generator = Generator(
        batch_size=BATCH_SIZE,
        input_exprs=input_exprs['train'],
        output_exprs=output_exprs['train'],
        reverse = REVERSE,
        **data_character
    )
    valid_generator = Generator(
        batch_size=BATCH_SIZE,
        input_exprs=input_exprs['valid'],
        output_exprs=output_exprs['valid'],
        reverse=REVERSE,
        **data_character
    )

    # Get RNN model
    rnn = RNN_S2S(
        units=latent_dim,
        num_input_tokens=num_input_tokens,  
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        rnn_layers=1
    )
    model = rnn.get_model()

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
        loss='categorical_crossentropy',
        optimizer=adam,
        metrics=['accuracy']
    )
    model.summary()
    # exit()

    es = EarlyStopping(monitor="loss", patience=10)

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
        use_multiprocessing=False,
        callbacks=[lr, es],
        # callbacks=None,
        validation_data=valid_generator,
        workers=cpu_count//2,
        verbose=1
    )

    # create and save plot of losses
    save_path = "../rnn_save/"

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
    # plt.ylim([0, 1])
    plt.grid(True, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path + "losses.png", format="png")
    plt.savefig(save_path + "losses.pdf", format="pdf")

    # create and save plot of evaluation metrics
    plt.figure()
    plt.plot(train_hist.history["accuracy"], color="b", label="train")
    plt.plot(
        train_hist.history["val_accuracy"],
        color="r",
        label="valid"
    )
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(loc="best")
    # plt.ylim([0, 1])
    plt.grid(True, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path + "accuracy.png", format="png")
    plt.savefig(save_path + "accuracy.pdf", format="pdf")


if __name__ == "__main__":
    main()
