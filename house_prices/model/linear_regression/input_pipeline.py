import tensorflow as tf


def input_fn(data_dataframe, target_dataframe,
             batch_size, num_epochs, should_shuffle):
    return tf.estimator.inputs.pandas_input_fn(
        x=data_dataframe,
        y=target_dataframe,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=should_shuffle)


def train_input_fn(data_dataframe, target_dataframe,
                   batch_size, num_epochs, should_shuffle):
    return input_fn(data_dataframe, target_dataframe,
                    batch_size, num_epochs, should_shuffle)


def validation_input_fn(data_dataframe, target_dataframe, batch_size):
    return input_fn(data_dataframe, target_dataframe, batch_size, 1, False)


def test_input_fn(data_dataframe):
    batch_size = data_dataframe.shape[0]
    return input_fn(data_dataframe, None, batch_size, 1, False)
