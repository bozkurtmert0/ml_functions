import numpy as np

# MASE implementation
def mean_absolute_scaled_error(y_true, y_pred):
  """
  Implement MASE (assuming no seasonality of data).
  yani veride kış veya yaz gibi belirli zamanlarıda veride etkisi yok
  """
  mae = tf.reduce_mean(tf.abs(y_true-y_pred))

  # Find MAE of naive forecast (no seasonality)
  mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1])) # our seasonality is 1 day (hence the shift of 1)

  return mae / mae_naive_no_season
  
def evaluate_preds(y_true, y_pred):
  #float32 datatype 
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  
  mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
  mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
  rmse = tf.sqrt(mse)
  mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
  mase = mean_absolute_scaled_error(y_true, y_pred)

  return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy(),
          "mase": mase.numpy()}  



def make_windows(x, window_size=WINDOW_SIZE, horizon=HORIZON):
  """

  full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
  len(full_windows), len(full_labels)
  
  """
  window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
  
  window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T 
  #print(f"Window indexes:\n {window_indexes, window_indexes.shape}")

  windowed_array = x[window_indexes]
  #print(windowed_array)
  
  windows, labels = get_labelled_windows(windowed_array, horizon=horizon)
  return windows, labels

def get_labelled_windows(x, horizon=HORIZON):
  """
  windowed data için label oluşturacağız
  test_window, test_label = get_labelled_windows(tf.expand_dims(tf.range(8), axis=0))
  print(f"Window: {tf.squeeze(test_window).numpy()} -> Label: {tf.squeeze(test_label).numpy()}")
   horizon=1
  Input: [0, 1, 2, 3, 4, 5, 6, 7] -> Output: ([0, 1, 2, 3, 4, 5, 6], [7])
  
  """
  return x[:, :-horizon], x[:, -horizon:]


def make_train_test_splits(windows, labels, test_split=0.2):
  """
  train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
  len(train_windows), len(test_windows), len(train_labels), len(test_labels)
  
  """
  split_size = int(len(windows) * (1-test_split)) 
  train_windows = windows[:split_size]
  train_labels = labels[:split_size]
  test_windows = windows[split_size:]
  test_labels = labels[split_size:]
  return train_windows, test_windows, train_labels, test_labels

def evaluate_preds(y_true, y_pred):
  # Make sure float32 datatype (for metric calculations)
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  # Calculate various evaluation metrics 
  mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
  mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
  rmse = tf.sqrt(mse)
  mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
  mase = mean_absolute_scaled_error(y_true, y_pred)

  #bu blok ile eger dimension sayisi 0 dan buyukse onu 0 dimensiona dusurecegiz
  if mae.ndim > 0:
    mae = tf.reduce_mean(mae)
    mse = tf.reduce_mean(mse)
    rmse = tf.reduce_mean(rmse)
    mape = tf.reduce_mean(mape)
    mase = tf.reduce_mean(mase)

  return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy(),
          "mase": mase.numpy()}

