import tensorflow as tf
import numpy as np
import torch
from keras.models import Sequential  # for creating a linear stack of layers for our Neural Network
from keras import Input  # for instantiating a keras tensor
from keras.layers import Bidirectional, GRU, RepeatVector, Dense, \
        TimeDistributed  # for creating layers inside the Neural Network
from tensorflow.keras.preprocessing.sequence import pad_sequences
def training(train_x,total_costs_train,max_length):

    # Pad the sequences
    train_x = pad_sequences(train_x, maxlen=max_length, padding='post', truncating='post', dtype='float32')
    train_x_tensors = []
    for i in range(0, len(train_x)):
        train_x_tensors.append(tf.convert_to_tensor(train_x[i]))
    train_x_tensors = tf.convert_to_tensor(train_x_tensors)

    total_costs_train = torch.stack(total_costs_train)
    total_costs_train_tensors = []
    for i in range(0, len(total_costs_train)):
        total_costs_train_tensors.append(tf.convert_to_tensor(total_costs_train[i]))
    total_costs_train_tensors = tf.convert_to_tensor(total_costs_train_tensors)

    train_x_tensors = tf.convert_to_tensor(train_x_tensors, dtype=tf.float32)
    # Reshape the input data
    train_x_tensors = np.reshape(train_x_tensors, (train_x_tensors.shape[0], 1, train_x_tensors.shape[1]))

    # Specify the input shape based on a single input sample
    input_shape = (1, train_x.shape[1])
    #=====================================================================================================
    #============================ TRAINING ================================================================
    #######################################################################################################

    ##### Step 3 - Specify the structure of a Neural Network
    model = Sequential(name="GRU-Model")  # Model
    model.add(Input(shape=input_shape,
                    name='Input-Layer'))  # Input Layer - need to speicfy the shape of inputs
    model.add(Bidirectional(GRU(units=32, activation='tanh', recurrent_activation='sigmoid', stateful=False),
                            name='Hidden-GRU-Encoder-Layer'))  # Encoder Layer
    model.add(RepeatVector(train_x.shape[1], name='Repeat-Vector-Layer'))  # Repeat Vector
    model.add(Bidirectional(
        GRU(units=32, activation='tanh', recurrent_activation='sigmoid', stateful=False, return_sequences=True),
        name='Hidden-GRU-Decoder-Layer'))  # Decoder Layer
    model.add(TimeDistributed(Dense(units=1, activation='linear'), name='Output-Layer'))  # Output Layer, Linear(x) = x

    ##### Step 4 - Compile the model
    model.compile(optimizer='adam',  # default='rmsprop', an algorithm to be used in backpropagation
                  loss='mean_squared_error',
                  # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
                  metrics=['MeanSquaredError', 'MeanAbsoluteError'],
                  # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance.
                  loss_weights=None,
                  # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
                  weighted_metrics=None,
                  # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
                  run_eagerly=None,
                  # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
                  steps_per_execution=None
                  # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
                  )

    ##### Step 5 - Fit the model on the dataset
    history = model.fit(train_x_tensors,  # input data
                        total_costs_train_tensors,  # target data
                        batch_size=1,
                        # Number of samples per gradient update. If unspecified, batch_size will default to 32.
                        epochs=50,
                        # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
                        verbose=1,
                        # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
                        callbacks=None,
                        # default=0.0, Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.
                        # validation_data=(X_test, y_test), # default=None, Data on which to evaluate the loss and any model metrics at the end of each epoch.
                        shuffle=True,
                        # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
                        class_weight=None,
                        # default=None, Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
                        sample_weight=None,
                        # default=None, Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).
                        initial_epoch=0,
                        # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
                        # steps_per_epoch=None,
                        # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
                        validation_steps=None,
                        # Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.
                        # validation_batch_size=None,
                        # Integer or None, default=None, Number of samples per validation batch. If unspecified, will default to batch_size.
                        validation_freq=10,
                        # default=1, Only relevant if validation data is provided. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs.
                        max_queue_size=10,
                        # default=10, Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
                        workers=1,
                        # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
                        use_multiprocessing=True,
                        # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False.
                        )
    #
    # # Save the trained model
    model.save('Benefit_Estimation_Model/Encoder/model(16245Q)')
    return max_length