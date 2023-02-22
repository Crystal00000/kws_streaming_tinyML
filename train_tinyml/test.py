# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test utility functions for accuracy evaluation."""
import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import kws_streaming.data.input_data as input_data
from kws_streaming.layers import modes
from kws_streaming.models import models
from kws_streaming.models import utils
from kws_streaming.train_tinyml import inference

from kws_streaming.dataloader import loadCSV
from kws_streaming.dataloader import IEGM_DataGenerator_test
from kws_streaming.dataloader import FB, count, convertmax, ACC


def tf_non_stream_model_accuracy(
    flags,
    best_model_num,
    folder, # tf
    time_shift_samples=0,
    weights_name='best_weights',
    accuracy_name='tf_non_stream_model_accuracy.txt'):
  """Compute accuracy of non streamable model using TF.

  Args:
      flags: model and data settings
      folder: folder name where accuracy report will be stored
      time_shift_samples: time shift of audio data it will be applied in range:
        -time_shift_samples...time_shift_samples
        We can use non stream model for processing stream of audio.
        By default it will be slow, so to speed it up
        we can use non stream model on sampled audio data:
        for example instead of computing non stream model
        on every 20ms, we can run it on every 200ms of audio stream.
        It will reduce total latency by 10 times.
        To emulate sampling effect we use time_shift_samples.
      weights_name: file name with model weights
      accuracy_name: file name for storing accuracy in path + accuracy_name
  Returns:
    accuracy
  """
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)
  tf.keras.backend.set_learning_phase(0)
  # flags.batch_size = 100  # set batch size for inference
  flags.batch_size = 1
  model = models.MODELS[flags.model_name](flags)
  model.load_weights(os.path.join(flags.train_dir, weights_name, 'model_F' + str(best_model_num) + '.hdf5'))
  model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=flags.optimizer_epsilon), loss='categorical_crossentropy', metrics=['accuracy'])
  
  test_csv_data = loadCSV('/home/109700045/tinyml/kws_streaming/data_indices/test_indice.csv')
  partition, labels = {}, {}
  partition['test'] = []

  for k, v in test_csv_data.items():
    partition['test'].append(k)
    labels[k] = v[0]
  test_dataset = IEGM_DataGenerator_test(partition['test'], labels, batch_size =flags.batch_size, shuffle=True, size=1250)

  x_test = test_dataset[0][0]
  x_test = np.expand_dims(x_test, axis = 2)
  x_test = np.expand_dims(x_test, axis = 3)
  y_test = test_dataset[0][1]

  # evaluate the model & compute scores
  pred = model.predict(x_test)
  predict = convertmax(pred)
  list = count(predict, y_test)
  fb = FB(list)
  _, baseline_model_accuracy = model.evaluate(x_test, y_test)
  print("FB: ", fb)
  print("ACC: ", baseline_model_accuracy)

  logging.info('TF Final test accuracy on non stream model = %.2f%%',
               (baseline_model_accuracy))

  path = os.path.join(flags.train_dir, folder)
  if not os.path.exists(path):
    os.makedirs(path)

  fname_summary = 'model_summary_non_stream'
  utils.save_model_summary(model, path, file_name=fname_summary + '.txt')

  tf.keras.utils.plot_model(
      model,
      to_file=os.path.join(path, fname_summary + '.png'),
      show_shapes=True,
      expand_nested=True)

  with open(os.path.join(path, accuracy_name), 'wt') as fd:
    fd.write('%f %f' % (baseline_model_accuracy, fb))
  return baseline_model_accuracy


def tf_stream_state_internal_model_accuracy(
    flags,
    best_model_num,
    folder,
    weights_name='best_weights',
    accuracy_name='tf_stream_state_internal_model_accuracy_sub_set.txt',
    max_test_samples=1000):
  """Compute accuracy of streamable model with internal state using TF.

  Testign model with batch size 1 can be slow, so accuracy is evaluated
  on subset of data with size max_test_samples
  Args:
      flags: model and data settings
      folder: folder name where accuracy report will be stored
      weights_name: file name with model weights
      accuracy_name: file name for storing accuracy in path + accuracy_name
      max_test_samples: max number of test samples. In this mode model is slow
        with TF because of batch size 1, so accuracy is computed on subset of
        testing data
  Returns:
    accuracy
  """
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)

  logging.info('tf stream model state internal without state resetting'
               'between testing sequences')

  inference_batch_size = 1
  tf.keras.backend.set_learning_phase(0)
  flags.batch_size = inference_batch_size  # set batch size
  model = models.MODELS[flags.model_name](flags)
  model.load_weights(os.path.join(flags.train_dir, weights_name, 'model_F' + str(best_model_num) + '.hdf5'))
  model_stream = utils.to_streaming_inference(
      model, flags, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
  model_stream.save(os.path.join(flags.train_dir, folder,  'stream_internal.h5'), include_optimizer=False, save_format='h5')
  model_stream.save(os.path.join(flags.train_dir, folder,  'stream_internal_w_optimizer.h5'), include_optimizer=True, save_format='h5')
  #load data
  test_csv_data = loadCSV('/home/109700045/tinyml/kws_streaming/data_indices/test_indice.csv')
  partition, labels = {}, {}
  partition['test'] = []

  for k, v in test_csv_data.items():
    partition['test'].append(k)
    labels[k] = v[0]
  test_dataset = IEGM_DataGenerator_test(partition['test'], labels, batch_size =flags.batch_size, shuffle=True, size=1250)

  x_test = test_dataset[0][0]
  x_test = np.expand_dims(x_test, axis = 2)
  x_test = np.expand_dims(x_test, axis = 3)
  y_test = test_dataset[0][1]

  pred = np.zeros(shape = (5625 , 2))
  for i in range(0, 5625):
    input_data = np.expand_dims(x_test[i], axis = 0) #(1,1250,1,1)
    
    outputs = inference.run_stream_inference_classification(
        flags, model_stream, input_data)
    pred[i] = outputs

  predict = convertmax(pred)
  list = count(predict, y_test)
  fb = FB(list)
  print("FB = ", fb)
  acc = ACC(list)
  print("ACC = ", acc)
  logging.info(
      'TF Final test accuracy of stream model state internal = %.2f%%', (acc))

  path = os.path.join(flags.train_dir, folder)
  if not os.path.exists(path):
    os.makedirs(path)

  fname_summary = 'model_summary_stream_state_internal'
  utils.save_model_summary(model_stream, path, file_name=fname_summary + '.txt')

  tf.keras.utils.plot_model(
      model_stream,
      to_file=os.path.join(path, fname_summary + '.png'),
      show_shapes=True,
      expand_nested=True)

  with open(os.path.join(path, accuracy_name), 'wt') as fd:
    fd.write('%f %f' % (acc, fb))
  return acc

def tf_stream_state_external_model_accuracy(
    flags,
    best_model_num,
    folder,
    weights_name='best_weights',
    accuracy_name='stream_state_external_model_accuracy.txt',
    reset_state=False,
    max_test_samples=1000):
  """Compute accuracy of streamable model with external state using TF.

  Args:
      flags: model and data settings
      folder: folder name where accuracy report will be stored
      weights_name: file name with model weights
      accuracy_name: file name for storing accuracy in path + accuracy_name
      reset_state: reset state between testing sequences.
        If True - then it is non streaming testing environment: state will be
          reseted on every test and will not be transferred to another one (as
          it is done in real streaming).
      max_test_samples: max number of test samples. In this mode model is slow
        with TF because of batch size 1, so accuracy is computed on subset of
        testing data
  Returns:
    accuracy
  """
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)

  inference_batch_size = 1
  flags.batch_size = inference_batch_size  # set batch size
  tf.keras.backend.set_learning_phase(0)
  model = models.MODELS[flags.model_name](flags)
  model.load_weights(os.path.join(flags.train_dir, weights_name, 'model_F' + str(best_model_num) + '.hdf5'))
  model_stream = utils.to_streaming_inference(
      model, flags, modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE)
  model_stream.save(os.path.join(flags.train_dir, folder,  'stream_external.h5'), include_optimizer=False, save_format='h5')
  model_stream.save(os.path.join(flags.train_dir, folder,  'stream_external_w_optimizer.h5'), include_optimizer=True, save_format='h5')
  
  logging.info('tf stream model state external with reset_state %d',
               reset_state)
  
  test_csv_data = loadCSV('/home/109700045/tinyml/kws_streaming/data_indices/test_indice.csv')
  partition, labels = {}, {}
  partition['test'] = []

  for k, v in test_csv_data.items():
    partition['test'].append(k)
    labels[k] = v[0]
  test_dataset = IEGM_DataGenerator_test(partition['test'], labels, batch_size =flags.batch_size, shuffle=True, size=1250)

  x_test = test_dataset[0][0]
  x_test = np.expand_dims(x_test, axis = 2)
  x_test = np.expand_dims(x_test, axis = 3)
  y_test = test_dataset[0][1]

  inputs = []
  for s in range(len(model_stream.inputs)):
    inputs.append(np.zeros(model_stream.inputs[s].shape, dtype=np.float32))

  print("len(inputs) = ", len(inputs))

  pred = np.zeros(shape = (5625 , 2))
  for i in range(0, 5625):
    input_data = np.expand_dims(x_test[i], axis = 0) #(1,1250,1,1)
    
    if reset_state:
      for s in range(len(model_stream.inputs)):
        inputs[s] = np.zeros(model_stream.inputs[s].shape, dtype=np.float32)


    start = 0
    end = flags.data_shape[0] # 360
    # print("end = ", end)
    # print("input_data.shape", input_data.shape)
    # iterate over time samples with stride = window_stride_samples
    while end <= input_data.shape[1]:
      # get new frame from stream of data
      stream_update = input_data[:, start:end]
      # print("stream_update.shape = ", stream_update.shape)

      # update indexes of streamed updates
      start = end
      end = start + flags.data_shape[0]

      # set input audio data (by default input data at index 0)
      inputs[0] = stream_update

      # run inference
      outputs = model_stream.predict(inputs)

      # get output states and set it back to input states
      # which will be fed in the next inference cycle
      for s in range(1, len(model_stream.inputs)):
        inputs[s] = outputs[s]  

    pred[i] = outputs[0]
  
  predict = convertmax(pred)
  list = count(predict, y_test)
  fb = FB(list)
  print("FB = ", fb)
  acc = ACC(list)
  print("ACC = ", acc)
  logging.info(
      'TF Final test accuracy of stream model state external with reset_state %d = %.2f%%' % (reset_state, acc))

  path = os.path.join(flags.train_dir, folder)
  if not os.path.exists(path):
    os.makedirs(path)

  fname_summary = 'model_summary_stream_state_external'
  utils.save_model_summary(model_stream, path, file_name=fname_summary + '.txt')

  tf.keras.utils.plot_model(
      model_stream,
      to_file=os.path.join(path, fname_summary + '.png'),
      show_shapes=True,
      expand_nested=True)

  with open(os.path.join(path, accuracy_name), 'wt') as fd:
    fd.write('%f %f' % (acc, fb))
    
  
  return acc


def tflite_stream_state_external_model_accuracy(
    flags,
    folder,
    tflite_model_name = 'stream_state_external.tflite',
    accuracy_name = 'tflite_stream_state_external_model_accuracy.txt',
    reset_state = False):
  """Compute accuracy of streamable model with external state using TFLite.

  Args:
      flags: model and data settings
      folder: folder name where model is located
      tflite_model_name: file name with tflite model
      accuracy_name: file name for storing accuracy in path + accuracy_name
      reset_state: reset state between testing sequences.
        If True - then it is non streaming testing environment: state will be
          reseted in the beginning of every test sequence and will not be
          transferred to another one (as it is done in real streaming).
  Returns:
    accuracy
  """
  # load test data
  test_csv_data = loadCSV('/home/109700045/tinyml/kws_streaming/data_indices/test_indice.csv')
  partition, labels = {}, {}
  partition['test'] = []

  for k, v in test_csv_data.items():
    partition['test'].append(k)
    labels[k] = v[0]
  test_dataset = IEGM_DataGenerator_test(partition['test'], labels, batch_size =flags.batch_size, shuffle=True, size=1250)

  x_test = test_dataset[0][0]
  x_test = np.expand_dims(x_test, axis = 2)
  x_test = np.expand_dims(x_test, axis = 3)
  y_test = test_dataset[0][1]

  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)
  path = os.path.join(flags.train_dir, folder)

  logging.info('tflite stream model state external with reset_state %d',
               reset_state)

  interpreter = tf.lite.Interpreter(
      model_path=os.path.join(path, tflite_model_name))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
    
  pred = np.zeros(shape = (5625 , 2))  
  inputs = []
  for s in range(len(input_details)):
    inputs.append(np.zeros(input_details[s]['shape'], dtype=np.float32))
  

  for i in range(0, 5625):
    input_data = np.expand_dims(x_test[i], axis = 0)
    # interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))

    # if reset_state:#?
    
    # run inference
    # interpreter.invoke()
    # get output: classification
    #
    out_tflite = inference.run_stream_inference_classification_tflite(
          flags, interpreter, input_data, inputs)
    #
    # out_tflite = interpreter.get_tensor(output_details[0]['index'])
    pred[i] = out_tflite

  # print("*****pred.shape = ", pred.shape)
  predict = convertmax(pred)
  list = count(predict, y_test)
  # print("*****list:", list)
  fb = FB(list)
  print("FB = ", fb)
  acc = ACC(list)
  print("ACC = ", acc)

  logging.info('tflite Final test accuracy on non stream model = %.2f%%', (acc))

  with open(os.path.join(path, accuracy_name), 'wt') as fd:
    fd.write('%f %f' % (acc, fb))
  return acc

def tflite_non_stream_model_accuracy(
    flags,
    folder, # 'quantize_opt_for_size_tflite_non_stream'
    tflite_model_name='non_stream.tflite',
    accuracy_name='tflite_non_stream_model_accuracy.txt'):
  """Compute accuracy of non streamable model with TFLite.

  Model has to be converted to TFLite and stored in path+tflite_model_name
  Args:
      flags: model and data settings
      folder: folder name where model is located
      tflite_model_name: file name with tflite model
      accuracy_name: file name for storing accuracy in path + accuracy_name
  Returns:
    accuracy
  """
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)
  path = os.path.join(flags.train_dir, folder)


  interpreter = tf.lite.Interpreter(
      model_path=os.path.join(path, tflite_model_name))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  # print("*********input_details = ", input_details)
  output_details = interpreter.get_output_details()
  # print("*********output_details = ", output_details)

  # load test data
  test_csv_data = loadCSV('/home/109700045/tinyml/kws_streaming/data_indices/test_indice.csv')
  partition, labels = {}, {}
  partition['test'] = []

  for k, v in test_csv_data.items():
    partition['test'].append(k)
    labels[k] = v[0]
  test_dataset = IEGM_DataGenerator_test(partition['test'], labels, batch_size =flags.batch_size, shuffle=True, size=1250)

  x_test = test_dataset[0][0]
  x_test = np.expand_dims(x_test, axis = 2)
  x_test = np.expand_dims(x_test, axis = 3)
  y_test = test_dataset[0][1]
  
  pred = np.zeros(shape = (5625 , 2))
  for i in range(0, 5625):
    input_data = np.expand_dims(x_test[i], axis = 0)
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))

    # run inference
    interpreter.invoke()

    # get output: classification
    out_tflite = interpreter.get_tensor(output_details[0]['index'])
    pred[i] = out_tflite

  # print("*****pred.shape = ", pred.shape)
  predict = convertmax(pred)
  list = count(predict, y_test)
  # print("*****list:", list)
  fb = FB(list)
  print("FB = ", fb)
  acc = ACC(list)
  print("ACC = ", acc)

  logging.info('tflite Final test accuracy on non stream model = %.2f%%', (acc))

  with open(os.path.join(path, accuracy_name), 'wt') as fd:
    fd.write('%f %f' % (acc, fb))
  return acc

def convert_model_tflite(flags,
                         best_model_num,
                         folder, # opt_name + 'tflite_non_stream'
                         mode,
                         fname,
                         weights_name='best_weights',
                         optimizations=None): # [tf.lite.Optimize.DEFAULT] or none
  """Convert model to streaming and non streaming TFLite.

  Args:
      flags: model and data settings
      folder: folder where converted model will be saved
      mode: inference mode
      fname: file name of converted model
      weights_name: file name with model weights
      optimizations: list of optimization options
  """
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)
  tf.keras.backend.set_learning_phase(0)
  flags.batch_size = 1  # set batch size for inference
  model = models.MODELS[flags.model_name](flags)
  model.load_weights(os.path.join(flags.train_dir, weights_name, 'model_F' + str(best_model_num) + '.hdf5'))
  # model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=flags.optimizer_epsilon), loss='categorical_crossentropy', metrics=['accuracy'])
  
  # convert trained model to non streaming TFLite stateless
  # to finish other tests we do not stop program if exception happen here
  path_model = os.path.join(flags.train_dir, folder)
  if not os.path.exists(path_model):
    os.makedirs(path_model)
  try:
    with open(os.path.join(path_model, fname), 'wb') as fd:
      fd.write(
          utils.model_to_tflite(sess, model, flags, mode, path_model,
                                optimizations))
  except IOError as e:
    logging.warning('FAILED to write file: %s', e)
  except (ValueError, AttributeError, RuntimeError, TypeError) as e:
    logging.warning('FAILED to convert to mode %s, tflite: %s', mode, e)

def convert_model_saved(flags, folder, mode, best_model_num, weights_name='best_weights'):
  """Convert model to streaming and non streaming SavedModel.

  Args:
      flags: model and data settings
      folder: folder where converted model will be saved
      mode: inference mode
      weights_name: file name with model weights
  """
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)
  tf.keras.backend.set_learning_phase(0)
  flags.batch_size = 1  # set batch size for inference
  model = models.MODELS[flags.model_name](flags)
  # model.load_weights(checkpoint_filepath)
  
  
  
  model.load_weights(os.path.join(flags.train_dir, weights_name, 'model_F' + str(best_model_num) + '.hdf5'))

  path_model = os.path.join(flags.train_dir, folder)
  if not os.path.exists(path_model):
    os.makedirs(path_model)
  try:
    # convert trained model to SavedModel
    utils.model_to_saved(model, flags, path_model, mode)
  except IOError as e:
    logging.warning('FAILED to write file: %s', e)
  except (ValueError, AttributeError, RuntimeError, TypeError,
          AssertionError) as e:
    logging.warning('WARNING: failed to convert to SavedModel: %s', e)
