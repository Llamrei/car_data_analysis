#!/usr/bin/env python
# coding: utf-8

# Direct copy pasta of initial project's file


#TODO: Make it so that it has a running save of the best graph
#TODO: Use cross fold validation
#TODO: Auto-tuning regularization params
#TODO: Get weight decay working
"""
Follows 
https://www.tensorflow.org/tutorials/text/text_classification_rnn
and 
https://www.tensorflow.org/tutorials/text/classify_text_with_bert
"""
import glob
import os
import pickle as pkl
import shutil
from pathlib import Path
import random
import re
import sys
from datetime import datetime
import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
# For pre-trained embeddings:
import tensorflow_hub as hub
import tensorflow_text as text

train_start = datetime.now()
print(f"In Python {datetime.now()}")
print("\n\nDevices")
print(tf.config.list_physical_devices())
print("\n\n")


# How much it loads into memory for sampling - we're ok just loading everything at once as it isn't a lot of data
BUFFER_SIZE = 10000
# Batch for gradient averaging
BATCH_SIZE = 64
# Specify encoding of words
VOCAB_SIZE = 5000
START = train_start.strftime('%Y-%m-%d-%H%M')
OUTPUT_DIR = Path(sys.argv[1])
GROSS_SYNOPSES_PATH = sys.argv[2]
EXP_STR = sys.argv[3]
print(EXP_STR)
loss_fn, embedding_strat, network_architecture, reg_strat = EXP_STR.split('-')
losses = {
    'mse':tf.keras.losses.MeanSquaredError(name='mse'),
    'mae':tf.keras.losses.MeanAbsoluteError(name='mae'),
    'mape':tf.keras.losses.MeanAbsolutePercentageError(name='mape')
}
INPUT_KEY="desc"
OUTPUT_KEY="price"

penalty_strat = None
weight_decay = None
dropout = None
DROPOUT_RATE = 0.1
WEIGHT_DECAY_RATIO = 1e-3

if reg_strat == 'l1':
        penalty_strat = tf.keras.regularizers.L1
elif reg_strat == 'l2':
    penalty_strat = tf.keras.regularizers.L2
elif reg_strat == 'wd':
    weight_decay = True
elif reg_strat == 'do':
    dropout = True

if len(sys.argv) > 5:
    intra_threads = int(sys.argv[4])
    inter_threads = int(sys.argv[5])
else:
    intra_threads = 1
    inter_threads = 1

tf.config.threading.set_intra_op_parallelism_threads(
    intra_threads
)
tf.config.threading.set_inter_op_parallelism_threads(
    inter_threads
)

print("\n\nInter:")
print(tf.config.threading.get_inter_op_parallelism_threads())
print("\n\nIntra:")
print(tf.config.threading.get_intra_op_parallelism_threads())    

def plot_graphs(history, metric, prefix=''):
    if prefix:
        prefix = f"{prefix}-"
    
    fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
    
    color = "tab:red"
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(metric, color=color)
    ax1.plot(history.history[metric], color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  

    color = "tab:blue"
    ax2.set_ylabel(
        "val_" + metric, color=color
    )  # we already handled the x-label with ax1
    ax2.plot(history.history["val_" + metric], color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend([metric, "val_" + metric])
    plt.savefig(OUTPUT_DIR/f"{prefix}{metric}_history.png", dpi=300)


def split_data_into_input_and_output(data):
    """Take given data of format from scraper [link] and return the inputs and outputs seperated.

    Args:
        data (list): A numpy array/list of named tuples which contains entries for 'gross',
        'title', 'synopsis' and 'year'.
    """
    # Did this cause wanted to be certain about preserving order but i think numpy does that for free
    data_in, data_out = list(zip(*[((x[INPUT_KEY]), x[OUTPUT_KEY]) for x in data]))
    return np.array(data_in), np.array(data_out)


def clean_copy(data, min_length=10, max_length=250, min_earning=0,max_earning=np.exp(30)):
    cleaned_data = np.fromiter(
        (x for x in data if len(x[INPUT_KEY].split()) > min_length), dtype=data.dtype
    )
    print(
        f"Crushed {len(data)} to {len(cleaned_data)} after removing sub {min_length} word synopses"
    )
    old_len = len(cleaned_data)

    cleaned_data = np.fromiter(
        (x for x in cleaned_data if len(x[INPUT_KEY].split()) < max_length), dtype=data.dtype
    )
    print(
        f"Crushed {old_len} to {len(cleaned_data)} after removing super {max_length} word synopses"
    )
    old_len = len(cleaned_data)

    cleaned_data = np.fromiter(
        (x for x in cleaned_data if x[OUTPUT_KEY] > min_earning), dtype=data.dtype
    )
    print(
        f"Crushed {old_len} to {len(cleaned_data)} after removing sub {min_earning} gross"
    )
    old_len = len(cleaned_data)

    cleaned_data = np.fromiter(
        (x for x in cleaned_data if x[OUTPUT_KEY] < max_earning), dtype=data.dtype
    )
    print(
        f"Crushed {old_len} to {len(cleaned_data)} after removing super {max_earning} gross"
    )
    old_len = len(cleaned_data)

    return cleaned_data


def confusion_plot(lab, pred, name, ax):
    """
    Helper function to pile on scatter plots of labels and predictions that are real numbers
    marking a helpful black line for the truth

    lab = label
    pred = prediction

    needs a call to plt.show() or plt.savefig() after it cumulatively builds this
    """
    ax.scatter(lab, lab, label="truth", s=2, color="black")
    ax.scatter(lab, pred, label=name, s=2)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel("Truth")
    ax.set_ylabel("Prediction")

def data_split(data, valid_fraction, test_fraction, train_fraction=None):
    """
    Returns `data` split into (test_set, validation_set, training_set) where the 
    """
    
    if train_fraction is None:
        train_fraction = 1 - test_fraction - valid_fraction
    rng = np.random.default_rng()
    rng.shuffle(data)
    len_d = len(data)
    test_idx = int(len_d*test_fraction)
    valid_idx = test_idx + int(len_d*valid_fraction)
    return (data[:test_idx], data[test_idx:valid_idx], data[valid_idx:])


# ## Load, clean, shuffle, split and cast data into Tensorflow compatible format
raw_data = pkl.load(open(GROSS_SYNOPSES_PATH, "rb"))
data = clean_copy(raw_data)
test, _, train = data_split(raw_data, valid_fraction=0, test_fraction=0.15)

# Fraction of overall data
train_data_in, train_data_out = split_data_into_input_and_output(train)
# valid_data_in, valid_data_out = split_data_into_input_and_output(valid)
test_data_in, test_data_out = split_data_into_input_and_output(test)

# Make dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices((train_data_in, train_data_out))
test_dataset = tf.data.Dataset.from_tensor_slices((test_data_in, test_data_out))

# Prefetch parrallelising loading + execution (not huge so not necessary)
# Dont need padded_batch as we are encoding and padding later
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(5)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(5)

TRAIN_STEPS_PER_EPOCH = len(train_data_in) // BATCH_SIZE
# VALID_STEPS_PER_EPOCH = len(valid_data_in) // BATCH_SIZE
TEST_STEPS_PER_EPOCH = len(test_data_in) // BATCH_SIZE

def build_encoding(encoding_strat, text_input):
    if encoding_strat == 'tfidf':
        encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=VOCAB_SIZE,output_mode='tf-idf')
        encoder.adapt(train_dataset.map(lambda text, label: text))
        net = encoder(text_input)
        return net
    elif encoding_strat == 'learned':
        encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=VOCAB_SIZE,output_mode='int')
        encoder.adapt(train_dataset.map(lambda text, label: text))
        net = encoder(text_input)
        net = tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=64, mask_zero=True)(net)
        net = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(10,name='LSTM'),name='Bidirectional')(net)
        return net
    elif encoding_strat == 'bert':
        load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", options=load_options)
        print("\n\n\nHub resolve:")
        print(hub.resolve("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"))
        print("\n\n")

        # Step 1: tokenize batches of text inputs.
        tokenize = hub.KerasLayer(preprocessor.tokenize, name='tokenizer')
        tokenized_inputs = [tokenize(text_input), ]

        # Step 2 (optional): modify tokenized inputs.

        # Step 3: pack input sequences for the Transformer encoder.
        seq_length = 250  # We filter out anything more earlier
        bert_pack_inputs = hub.KerasLayer(
            preprocessor.bert_pack_inputs,
            arguments=dict(seq_length=seq_length), name='input_packer')  # Optional argument.
        encoder_inputs = bert_pack_inputs(tokenized_inputs)

        encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1", trainable=False, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        sequence_output = outputs["sequence_output"]
        net = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(10,name='LSTM'),name='Bidirectional')(sequence_output)
        return net
    else:
        raise ValueError(f"Invalid encoding strategy - given {encoding_strat}")



# ## Build model

def build_model(encoding_strat,model_strat):
    #TODO: Don't rely on global scope, should functionalise
    
    def regularize(scale):
        if penalty_strat:
            return {'activation':'relu','kernel_regularizer':penalty_strat(scale), 'bias_regularizer':penalty_strat(scale)}
        else:
            return {'activation':'relu'}
    
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    net = build_encoding(encoding_strat, text_input)
    if model_strat == 'core':
        scale = 2
        net = tf.keras.layers.Dense(64, name='hidden-1', **regularize(scale))(net)
        if dropout:
            net = tf.keras.layers.Dropout(DROPOUT_RATE)(net)
        net = tf.keras.layers.Dense(32, name='hidden-2', **regularize(scale))(net)
    elif model_strat == 'shallow_wide':
        scale = 1.5
        net = tf.keras.layers.Dense(2048, name='hidden-1', **regularize(scale))(net)
        if dropout:
            net = tf.keras.layers.Dropout(DROPOUT_RATE)(net)
    elif model_strat == 'deep_narrow':
        scale = 3
        net = tf.keras.layers.Dense(64, name='hidden-1', **regularize(scale))(net)
        if dropout:
            net = tf.keras.layers.Dropout(DROPOUT_RATE)(net)
        net = tf.keras.layers.Dense(32, name='hidden-2', **regularize(scale))(net)
        net = tf.keras.layers.Dense(16, name='hidden-3', **regularize(scale))(net)
    elif model_strat == 'deep_wide':
        scale = 3.5
        net = tf.keras.layers.Dense(128, name='hidden-1', **regularize(scale))(net)
        if dropout:
            net = tf.keras.layers.Dropout(DROPOUT_RATE)(net)
        net = tf.keras.layers.Dense(64, name='hidden-2', **regularize(scale))(net)
        net = tf.keras.layers.Dense(32, name='hidden-3', **regularize(scale))(net)
    else:
        raise ValueError(f"Invalid architecture - {model_strat}")
    net = tf.keras.layers.Dense(1, name='classifier')(net)
    return tf.keras.Model(text_input, net)

pre_trained_model = build_model(embedding_strat,network_architecture)

# #TODO: See if this improves results
# initial_learning_rate = 0.1
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=25*TRAIN_STEPS_PER_EPOCH,
#     decay_rate=0.1,
#     staircase=True)
if weight_decay:
    optimizer = tfa.optimizers.AdamW(1e-1,1e-1*WEIGHT_DECAY_RATIO)
else:
    optimizer = tf.keras.optimizers.Adam(0.1)

loss = losses.pop(loss_fn)
metrics = list(losses.values())

pre_trained_model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=metrics
)

# Document model to be used
with open(OUTPUT_DIR/'model_summary.txt', 'w') as f:
    pre_trained_model.summary()
    pre_trained_model.summary(print_fn=lambda x: f.write(f'{x}\n'), line_length=120)
    f.write(f'\n\n Optimizer:\n{optimizer.get_config()}')
    f.write(f'\n\n Loss:\n{loss.get_config()}')
    for metric in metrics:
        f.write(f'\n\n Metric:\n{metric.get_config()}')
    f.write(f"\n\n Reg strat: {reg_strat}")

checkpoint_dir = OUTPUT_DIR / "pre_trained_checkpoints"
checkpoint_path = checkpoint_dir / "{epoch}"
# Create a callback that saves the model's weights every other epoch
# NOTE: Important to delete the ones we aren't interested in!
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=2*TRAIN_STEPS_PER_EPOCH,
)
csv_logger = tf.keras.callbacks.CSVLogger(OUTPUT_DIR / 'training.csv',append=True)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=30, mode='min', restore_best_weights=False, verbose=1)

# class LearningRateLoggingCallback(tf.keras.callbacks.Callback):

#     def on_epoch_begin(self, epoch, logs=None):
#         lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
#         print(f'Learning rate {lr((epoch)*TRAIN_STEPS_PER_EPOCH)}')
# lr = LearningRateLoggingCallback()

# class AnimateConfusionCallback(tf.keras.callbacks.Callback):
#     def __init__(self):
#         super().__init__()
#         self.fig = plt.figure()
#         self.ax = self.fig.subplots()
#         # self.ax.plot() get round to plotting truth seperately
    
#     def on_epoch_end(self, epoch, logs=None):
#         predictions = dict()
#         predictions['test'] = self.model.predict(test_data_in)
#         predictions['train'] = self.model.predict(train_data_in)
#         confusion_plot(test_data_out, predictions['test'], 'test', self.fig)
#         confusion_plot(train_data_out, predictions['train'], 'train', self.fig)

train_start = datetime.now()
pre_trained_history = pre_trained_model.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    validation_steps=TEST_STEPS_PER_EPOCH,
    callbacks=[cp_callback,csv_logger, early_stopping],
    verbose=2,
)

print(f"Training took {datetime.now()-train_start}")

pred_train = pre_trained_model.predict(train_data_in)
pred_test = pre_trained_model.predict(test_data_in)
trained_mean = np.mean(train_data_out)
trained_median = np.median(train_data_out)

XMAX = max(max(pred_train), max(pred_test), max(train_data_out), max(test_data_out))

fig = plt.figure(figsize=(11, 8), dpi=300)
ax = fig.subplots()
ax.set_title(EXP_STR)
confusion_plot(train_data_out, pred_train, f"train", ax)
confusion_plot(test_data_out, pred_test, f"test", ax)
ax.hlines(y=trained_mean, xmin=0.0, xmax=XMAX, color='grey', linestyles='dashed')
ax.hlines(y=trained_median, xmin=0.0, xmax=XMAX, color='grey', linestyles='dotted')
fig.savefig(OUTPUT_DIR / "pretrained_confusion.png")

plot_graphs(pre_trained_history, "loss")
for key_left in losses:
    plot_graphs(pre_trained_history, key_left)


# Have a peek at how the best predictions looked
pre_trained_model.set_weights(early_stopping.get_best_weights())
pred_train = pre_trained_model.predict(train_data_in)
pred_test = pre_trained_model.predict(test_data_in)

fig = plt.figure(figsize=(11, 8), dpi=300)
ax = fig.subplots()
ax.set_title(EXP_STR)
confusion_plot(train_data_out, pred_train, f"train", ax)
confusion_plot(test_data_out, pred_test, f"test", ax)
ax.hlines(y=trained_mean, xmin=0.0, xmax=XMAX, color='grey', linestyles='dashed')
ax.hlines(y=trained_median, xmin=0.0, xmax=XMAX, color='grey', linestyles='dotted')
fig.savefig(OUTPUT_DIR / "best-pretrained_confusion.png")


# Dump all relevant data for future plots if needed
pkl.dump(train_data_in,open(OUTPUT_DIR/'train_data_in.pkl','wb'))
pkl.dump(train_data_out,open(OUTPUT_DIR/'train_data_out.pkl', 'wb'))
pkl.dump(test_data_in,open(OUTPUT_DIR/'test_data_in.pkl','wb'))
pkl.dump(test_data_out,open(OUTPUT_DIR/'test_data_out.pkl', 'wb'))

pkl.dump(pred_test,open(OUTPUT_DIR/'pred_test.pkl','wb'))
pkl.dump(pred_train,open(OUTPUT_DIR/'pred_train.pkl','wb'))

pkl.dump(pred_test,open(OUTPUT_DIR/'best-pred_test.pkl','wb'))
pkl.dump(pred_train,open(OUTPUT_DIR/'best-pred_train.pkl','wb'))

#PBS -lselect=1:ncpus=8:mem=48gb:ngpus=2:gpu_type=RTX6000
#PBS -J 1-81
