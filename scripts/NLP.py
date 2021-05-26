import datetime
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense

tf.config.threading.set_intra_op_parallelism_threads(
    7
)
tf.config.threading.set_inter_op_parallelism_threads(
    7
)


job_id = sys.argv[1]
array_idx = sys.argv[2]
data_file = Path(sys.argv[3]) / "deduped_data.csv"
save_dir = sys.argv[4]

data = pd.read_csv(data_file)
data = data[["id","desc","price"]]

# TODO: Make this a proper option - for now comment/uncomment
# # For generating indices 
# indices_dir = data_file.parent.parent / "ensemble_indices"
# for i in range(30):
#     train, test = train_test_split(data, test_size=0.3)
#     train["id"].to_csv(indices_dir / f"train_{i}.csv", index=False)
#     test["id"].to_csv(indices_dir / f"test_{i}.csv", index=False)

# # For individiual runs we want to regenerate
# train, test = train_test_split(data, test_size=0.3)
# # For ensemble trainings we want to make sure everything uses same data
file_idx = int(array_idx) // 6
print(f"Loading indices from {file_idx}")
indices_dir = data_file.parent.parent / "ensemble_indices"
train_ids = pd.read_csv(indices_dir/ f"train_{file_idx}.csv")
test_ids = pd.read_csv(indices_dir/ f"test_{file_idx}.csv")
assert len(pd.merge(train_ids, test_ids, on="id")) == 0
train = pd.merge(train_ids, data, on="id", validate="one_to_one")
test = pd.merge(test_ids, data, on="id", validate="one_to_one")

train["test"] = 0
test["test"] = 1

data = pd.concat((train, test))

def lower_strip_punc_and_no_nums(inputs):
    lowercase_inputs = tf.strings.lower(inputs)
    DEFAULT_STRIP_REGEX = r'[!"#$£%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']'
    inputs = tf.strings.regex_replace(lowercase_inputs, DEFAULT_STRIP_REGEX, "")
    inputs = tf.strings.regex_replace(inputs, r"[0-9]*", "")
    return inputs

VOCAB_SIZE = 5000

text_input = Input(shape=(), dtype=tf.string, name='text')
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='tf-idf', 
            standardize=lower_strip_punc_and_no_nums, 
            max_tokens=5000)
encoder.adapt(train["desc"].values)
net = encoder(text_input)
net = Dense(64, activation="relu")(net)
net = Dense(32, activation="relu")(net)
net = Dense(1, activation="relu")(net)

model = tf.keras.Model(text_input, net)

BATCH_SIZE = 64
STEPS_PER_EPOCH = len(train) // BATCH_SIZE
boundaries = [30*STEPS_PER_EPOCH, 50*STEPS_PER_EPOCH]
values = [0.001, 1e-5, 1e-8]
lr = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

loss = keras.losses.MeanSquaredError()
optim = tf.keras.optimizers.RMSprop(learning_rate=lr)

model.compile(loss=loss, optimizer=optim)

early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True)
save_dir = Path(save_dir)/ "NLP"
save_dir.mkdir(parents=True, exist_ok=True)
(save_dir / "training").mkdir(parents=True, exist_ok=True)
csv_logger = tf.keras.callbacks.CSVLogger(save_dir / f'training/training_{job_id}_{array_idx}.csv', append=True)

start = datetime.datetime.now()
history = model.fit(train["desc"].values, 
                    train["price"].values,
                    epochs=100,
                    batch_size=BATCH_SIZE,
                    validation_data=(test["desc"].values, test["price"].values),
                    callbacks=[early_stopping, csv_logger],
                    verbose=2,
                   )
end = datetime.datetime.now()

time_elapsed = (end - start).total_seconds()
data["prediction"] = model.predict(data["desc"])
data[["id", "price", "prediction", "test"]].to_csv(save_dir/f"{job_id}_{array_idx}_{int(time_elapsed)}.csv", index=False)

