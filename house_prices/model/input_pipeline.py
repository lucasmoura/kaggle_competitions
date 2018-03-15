import tensorflow as tf


def input_fn(data_dataframe, target_dataframe, batch_size, num_epochs, should_shuffle):
    return tf.estimator.inputs.pandas_input_fn(
        x=data_dataframe,
        y=target_dataframe,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=should_shuffle)


def train_input_fn(data_dataframe, target_dataframe, batch_size, num_epochs, should_shuffle):
    return input_fn(data_dataframe, target_dataframe, batch_size, num_epochs, should_shuffle)
