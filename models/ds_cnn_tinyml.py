from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers import stream
from kws_streaming.layers.compat import tf
import kws_streaming.models.model_utils as utils

def model(flags):

    input_shape = modes.get_tinyml_input_shape(flags, modes.Modes.TRAINING) # (1250, 1, 1)
    input = tf.keras.layers.Input(
            shape = input_shape,
            batch_size = flags.batch_size)
    print("model input shape : ", input.shape)
    net = input
    
    # max pooling
    net = tf.keras.layers.MaxPooling2D(
            pool_size = (5,1), # N 'HW' C
            strides = (5,5))(net)
    print("after maxpool : ", net.shape)
    # conv1
    net = stream.Stream( 
            cell = tf.keras.layers.Conv2D( # (11, 250, 1, 1) ->  (11, (250-6+1)/2, 1, 1) = (11, 123, 1, 1)
            kernel_size = (6,1),
            filters = 3,
            padding = 'valid', # causal: limited to Conv1D       # same: invalid
            strides = (2,2)),
            use_one_step = False)(net)

    net = tf.keras.layers.BatchNormalization(
            momentum = 0.1,
            epsilon = 1e-5)(net)
    
    net = tf.keras.layers.Activation('relu')(net)

    # ds conv1
    net = stream.Stream(
          cell = tf.keras.layers.DepthwiseConv2D(
          kernel_size = (5,1),
          depth_multiplier = 1,
          padding = 'valid',
          strides = (2,2)),
          use_one_step = False)(net)
    
    net = tf.keras.layers.BatchNormalization(
          momentum = 0.1,
          epsilon = 1e-5)(net)
    
    net = tf.keras.layers.Activation('relu')(net)

    # ds conv2
    net = stream.Stream(
          cell = tf.keras.layers.DepthwiseConv2D(
          kernel_size = (4,1),
          depth_multiplier = 2,
          padding = 'valid',
          strides = (1,1)),
          use_one_step = False)(net)
    
    net = tf.keras.layers.BatchNormalization(
          momentum = 0.1,
          epsilon = 1e-5)(net)
    
    net = tf.keras.layers.Activation('relu')(net)

    # avg pooling
    net = stream.Stream(
        cell = tf.keras.layers.AveragePooling2D(
        pool_size = (3,1),
        strides = (3,1),
        padding = 'valid'),
        use_one_step = False)(net)


    # ds conv3
    net = stream.Stream(
          cell = tf.keras.layers.DepthwiseConv2D(
          kernel_size = (4,1),
          depth_multiplier = 2,
          padding = 'valid',
          strides = (2,2)),
          use_one_step = False)(net)
    
    net = tf.keras.layers.BatchNormalization(
          momentum = 0.1,
          epsilon = 1e-5)(net)
    
    net = tf.keras.layers.Activation('relu')(net)

    # ds conv4
    net = stream.Stream(
          cell = tf.keras.layers.DepthwiseConv2D(
          kernel_size = (4,1),
          depth_multiplier = 2,
          padding = 'valid',
          strides = (3,3)),
          use_one_step = False)(net)
    
    net = tf.keras.layers.BatchNormalization(
          momentum = 0.1,
          epsilon = 1e-5)(net)
    
    net = tf.keras.layers.Activation('relu')(net)

    # dropout and FC
    net = tf.keras.layers.Dropout(rate = 0.5)(net)
    net = stream.Stream(cell=tf.keras.layers.Flatten())(net)
    net = tf.keras.layers.Dense(units=10, activation='relu')(net)
    net = tf.keras.layers.Dense(units=2, activation='softmax')(net)
  
    return tf.keras.Model(input, net)

