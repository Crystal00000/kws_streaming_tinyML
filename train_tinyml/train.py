import os.path
import pprint
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
import kws_streaming.data.input_data as input_data
from kws_streaming.models import models
from kws_streaming.models import utils

from kws_streaming.dataloader import loadCSV
from kws_streaming.dataloader import IEGM_DataGenerator, IEGM_DataGenerator_test
from kws_streaming.dataloader import FB, count, convertmax
from sklearn.model_selection import StratifiedKFold


def train(flags):
  seed = 7
  np.random.seed(seed)
  EPOCH = 1
  K = 5

  flags.training = True

  # Set the verbosity based on flags (default is INFO, so we see all messages)
  logging.set_verbosity(flags.verbosity)

  # Start a new TensorFlow session.
  tf.reset_default_graph()

  # allow_soft_placement solves issue with
  # "No device assignments were active during op"
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)
  
  logging.info(flags)
  model = models.MODELS[flags.model_name](flags)
  logging.info(model.summary())

  # save model summary
  utils.save_model_summary(model, flags.train_dir)

  # save model and data flags
  with open(os.path.join(flags.train_dir, 'flags.txt'), 'wt') as f:
    pprint.pprint(flags, stream=f)

  loss = 'categorical_crossentropy'
  optimizer = tf.keras.optimizers.Adam(epsilon=flags.optimizer_epsilon)
  

  train_writer = tf.summary.FileWriter(
      os.path.join(flags.summaries_dir, 'train'), sess.graph)
  validation_writer = tf.summary.FileWriter(
      os.path.join(flags.summaries_dir, 'validation'))

  tf.train.write_graph(sess.graph_def, flags.train_dir, 'graph.pbtxt')


  # configure checkpointer
  checkpoint_directory = os.path.join(flags.train_dir, 'restore/')
  checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

  sess.run(tf.global_variables_initializer())
  status.initialize_or_restore(sess)

  train_csv_data = loadCSV('/home/109700045/tinyml/kws_streaming/data_indices/train_indice.csv')
  test_csv_data = loadCSV('/home/109700045/tinyml/kws_streaming/data_indices/test_indice.csv')
  partition, labels = {}, {}
  partition['train'] = []
  partition['test'] = []
  for k, v in train_csv_data.items():
    partition['train'].append(k)
    labels[k] = v[0]
        
  for k, v in test_csv_data.items():
    partition['test'].append(k)
    labels[k] = v[0]
      
  train_dataset = IEGM_DataGenerator(partition['train'], labels, batch_size = flags.batch_size, shuffle=True, size=1250)
  test_dataset = IEGM_DataGenerator_test(partition['test'], labels, batch_size =flags.batch_size, shuffle=True, size=1250)
    
  x_train = train_dataset[0][0] # 24588, 1250
  x_train = np.expand_dims(x_train, axis = 2) # 24588, 1250, 1
  x_train = np.expand_dims(x_train, axis = 3) # 24588, 1250, 1, 1
  x_train = np.delete(x_train, [0, 1, 2], 0) # 24585, 1250, 1, 1
  print("x_train shape : ", x_train.shape)
  y_train_raw = train_dataset[0][1]
  y_train_raw = np.delete(y_train_raw, [0, 1, 2], 0)

  x_test = test_dataset[0][0]
  x_test = np.expand_dims(x_test, axis = 2)
  x_test = np.expand_dims(x_test, axis = 3)
  y_test = test_dataset[0][1]

  y_train = np.ndarray(shape=(24585, 1), dtype=int)
  for i, _ in enumerate(y_train_raw):
      if y_train_raw[i][0] == 1:
          y_train[i] = 0 
      else:
          y_train[i] = 1

  # augmentation
  rng = np.random.default_rng()
  factor = rng.uniform(low = 0.8, high = 1.2, size=(24585, 1250, 1, 1))
  x_train = x_train * factor

  print("Epoch: ", EPOCH)
  print("K = ", K)
  kfold = StratifiedKFold(n_splits = K, shuffle = True, random_state = seed)
  cvscores = []
  fbs = []
  i = 1
  best_fb = 0
  os.makedirs(os.path.join(flags.train_dir, 'best_weights'))

  for train, test in kfold.split(x_train, y_train):
    print(f"fold {i} out of ", K)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    checkpoint_filepath = os.path.join(flags.train_dir, 'best_weights/model_F%d.hdf5' %(i))
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_filepath,
        save_weights_only = True,
        monitor = 'val_loss',
        mode = 'min',
        verbose = 1,
        save_best_only = True)

    history = model.fit(
        x_train[train], 
        y_train_raw[train], 
        validation_data = (x_train[test], y_train_raw[test]),
        batch_size = flags.batch_size,
        epochs = EPOCH,
        verbose = 0,
        callbacks=[model_checkpoint_callback]
        )
    
    # load best weights
    model.load_weights(checkpoint_filepath)

    # evaluate the model & compute scores
    pred = model.predict(x_test)
    predict = convertmax(pred)
    list = count(predict, y_test)
    fb = FB(list)
    _, baseline_model_accuracy = model.evaluate(x_test, y_test)
    print("acc: ", baseline_model_accuracy)
    print("fb score: ", fb)
    fbs.append(fb)
    cvscores.append(fb * 100)

    # save the best model for this fold
    fb *= 10000
    # best_model_filepath = os.path.join(flags.train_dir, 'trained_models/model_best_%d_F%d.h5' %(fb, i))
    # model.save(best_model_filepath, include_optimizer=False, save_format='h5')
    best_model_filepath = os.path.join(flags.train_dir, 'trained_models/model_best_%d_F%d.h5' %(fb, i))
    model.save(best_model_filepath, include_optimizer=False, save_format='h5')
    best_model_filepath = os.path.join(flags.train_dir, 'trained_models/model_best_%d_F%d_w_optimizer.h5' %(fb, i))
    model.save(best_model_filepath, include_optimizer=True, save_format='h5')

    print("model_best_%d_F%d.h5 saved!" %(fb, i))

    if best_fb < fb:
       best_fb = fb
       best_model_num = i
    i += 1

  # print cross validation result
  print("fbs: ", fbs)
  print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
  return best_model_num