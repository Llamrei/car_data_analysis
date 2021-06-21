import datetime
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras as K
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_addons as tfa


tf.config.threading.set_intra_op_parallelism_threads(
    7
)
tf.config.threading.set_inter_op_parallelism_threads(
    7
)

job_id = sys.argv[1]
array_idx = sys.argv[2]
data_dir = Path(sys.argv[3])
data_file = data_dir / "deduped_data.csv"
save_dir = sys.argv[4]

SHUFFLE_BUFFER = 1000
BATCH_SIZE = 64
HEIGHT = 256
WIDTH = 256

data = pd.read_csv(data_file)
data = data[["id", "price"]]

image_names = pd.DataFrame(data = [(x.stem, str(x)) for x in (data_dir / "images").glob("*.jpeg")], columns=["id", "filepath"])

data = pd.merge(data, image_names, on="id")
assert(len(data)==len(image_names))

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

train_ds = tf.data.Dataset.from_tensor_slices((train["filepath"].values, train["price"].values))
test_ds = tf.data.Dataset.from_tensor_slices((test["filepath"].values, test["price"].values))

def retrieve_image_tensor(filepath, price):
    im = tf.io.read_file(filepath)
    im = tf.image.decode_jpeg(im)
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = tf.image.resize(im, [HEIGHT, WIDTH])
    return im, price

train_ds = train_ds.map(retrieve_image_tensor).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
test_ds = test_ds.map(retrieve_image_tensor).batch(BATCH_SIZE)

# For retaining order so predictions can be tied to images
data_ds = tf.data.Dataset.from_tensor_slices((data["filepath"].values, data["price"].values))
i = 0
for x in data_ds.take(len(data)):
    assert x[1] == data["price"].iloc[i]
    i += 1
data_ds = data_ds.map(retrieve_image_tensor).batch(1)
module = hub.KerasLayer("https://tfhub.dev/google/bit/m-r50x1/1")
model = K.Sequential([
    K.Input(shape=(HEIGHT,WIDTH,3)),
    module,
    K.layers.Dense(64, activation="relu"),
    K.layers.Dense(32, activation="relu"),
    K.layers.Dense(1)
])

# print(model.predict(data_ds.take(1)))

STEPS_PER_EPOCH = len(train) // BATCH_SIZE
boundaries = [30*STEPS_PER_EPOCH, 50*STEPS_PER_EPOCH]
values = [0.001, 1e-5, 1e-8]
lr = K.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

loss = K.losses.MeanSquaredError()
optim = tf.keras.optimizers.RMSprop()

model.compile(loss=loss, optimizer=optim)

early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True)
save_dir = Path(save_dir)/ "CNN"
save_dir.mkdir(parents=True, exist_ok=True)
(save_dir / "training").mkdir(parents=True, exist_ok=True)
csv_logger = tf.keras.callbacks.CSVLogger(save_dir / f'training/training_{job_id}_{array_idx}.csv', append=True)
almost_day_stop = tfa.callbacks.TimeStopping(
    seconds: int = 22*60*60, # 22hrs
    verbose: int = 0
)

start = datetime.datetime.now()
history = model.fit(train_ds, 
                    epochs=30,
                    validation_data=test_ds,
                    callbacks=[early_stopping, csv_logger, almost_day_stop],
                    verbose=2,
                   )
end = datetime.datetime.now()

time_elapsed = (end - start).total_seconds()


data["prediction"] = model.predict(data_ds)
data[["id", "price", "prediction", "test"]].to_csv(save_dir/f"{job_id}_{array_idx}_{int(time_elapsed)}.csv", index=False)
