import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import sys
import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split

job_id = sys.argv[1]
array_idx = sys.argv[2]
data_file = Path(sys.argv[3]) / "deduped_data.csv"
save_dir = sys.argv[4]

data = pd.read_csv(data_file)
data["age"] = 2021 - data["manufactured.year"]
del data["manufactured.year"]
factors = set(data["make"].values)

train, test = train_test_split(data, test_size=0.3)
train["test"] = 0
test["test"] = 1

data = pd.concat((train, test))

non_make_cols = ["mileage","age", "engine.size","owners"]
make_col = ["make"]
price_col = ["price"]

train_makes, train_prices, train_input = train[make_col].values.flatten(), train[price_col].values, train[non_make_cols].values
test_makes, test_prices, test_input = test[make_col].values.flatten(), test[price_col].values, test[non_make_cols].values

encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='tf-idf', split=None)
encoder.adapt(train["make"].values)
make_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="make")
x = encoder(make_input)

reg = keras.regularizers.L2(l2=0.2)
init = keras.initializers.GlorotUniform()

other_input = keras.Input(shape=(4,), name="other")
x = layers.concatenate([x, other_input])
x = layers.BatchNormalization()(x)
x = layers.Dense(64)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, 
                 activation="relu", 
                 kernel_regularizer=reg,
                kernel_initializer=init
                )(x)
x = layers.BatchNormalization()(x)
# x = layers.Dense(128, 
#                  activation="relu", 
#                  kernel_regularizer=reg,
#                 kernel_initializer=init
#                 )(x)
# x = layers.BatchNormalization()(x)
# x = layers.Dense(64, 
#                  activation="relu", 
#                  kernel_regularizer=reg,
#                 kernel_initializer=init
#                 )(x)
# x = layers.BatchNormalization()(x)
x = layers.Dense(1)(x)
# x = layers.Activation(round_to(500))(x)
net = layers.LeakyReLU(1, name="price")(x)

model = keras.Model([make_input,other_input], net)
BATCH_SIZE = 64
STEPS_PER_EPOCH = len(train_makes) // BATCH_SIZE
boundaries = [30*STEPS_PER_EPOCH, 50*STEPS_PER_EPOCH]
values = [0.001, 1e-5, 1e-8]
lr = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

loss = keras.losses.MeanSquaredError()
optim = tf.keras.optimizers.RMSprop(learning_rate=lr)

model.compile(loss=loss, optimizer=optim)

early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True)
save_dir = Path(save_dir)/ "NN"
save_dir.mkdir(parents=True, exist_ok=True)
(save_dir / "training").mkdir(parents=True, exist_ok=True)
csv_logger = tf.keras.callbacks.CSVLogger(save_dir / f'training/training_{job_id}_{array_idx}.csv', append=True)

start = datetime.datetime.now()
history = model.fit([train_makes, train_input], 
                    train_prices,
                    epochs=100,
                    batch_size=BATCH_SIZE,
                    validation_data=([test_makes, test_input], test_prices),
                    callbacks=[early_stopping]
                   )
end = datetime.datetime.now()

time_elapsed = (end - start).total_seconds()
data["prediction"] = model.predict([data[make_col], data[non_make_cols]])
data[["id", "price", "prediction", "test"]].to_csv(save_dir/f"{job_id}_{array_idx}_{int(time_elapsed)}.csv", index=False)

