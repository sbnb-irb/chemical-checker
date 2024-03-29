import logging
import chemicalchecker
logging.log(logging.INFO, '{:<10} : {}'.format(
    'chemicalchecker', chemicalchecker.__path__))
logging.log(logging.INFO, '********** TEST START **********')

# print system info
import os
import sys
logging.log(logging.INFO, '{:<10} : {}'.format(
    'python', os.path.dirname(sys.executable)))
logging.log(logging.INFO, sys.version)

# print node info
import platform
for k, v in platform.uname()._asdict().items():
    logging.log(logging.INFO, '{:<10} : {}'.format(k, v))

# test tensorflow
import tensorflow as tf
logging.log(logging.INFO, '{:<10} : {}'.format('tensorflow', tf.__version__))

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# test faiss
import faiss
import numpy as np
logging.log(logging.INFO, '{:<10} : {}'.format('faiss', faiss.__version__))
logging.log(logging.INFO, '{:<10} : {}'.format('numpy', np.__version__))
faiss.Kmeans(10, 20).train(np.random.rand(1000, 10).astype(np.float32))

logging.log(logging.INFO, '********** TEST DONE **********')
