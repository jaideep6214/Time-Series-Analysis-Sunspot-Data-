# Time-Series-Analysis-Sunspot-Data-
Sunspots are temporary phenomena on the Sun's photosphere that appear as spots darker than the surrounding areas. They are regions of reduced surface temperature caused by concentrations of magnetic field flux that inhibit convection. Sunspots usually appear in pairs of opposite magnetic polarity. Their number varies according to the approximately 11-year solar cycle. 

Using Conv2D, Bidirectional LSTM and Dense Layer i was able to achieve 14.263876 MAE, below is my model's architecture.
model4 = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(64, 5, strides=1, padding="same", activation="relu", input_shape=[None, 1]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 200.0)
])
model4.compile(loss=tf.keras.losses.Huber(),optimizer='adam',metrics=["mae"])
history = model4.fit(train_set,epochs=100)
