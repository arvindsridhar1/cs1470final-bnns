import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):

    # model with 1 dense layer that gives a softmax probability over 10 possible classes
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        return self.dense(inputs) 

def load_mnist():
    ''' 
    Loads the MNIST data, reshapes it to one directional vectors, 
    normalizes data & sets type, and asserts reshape
    '''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (60000, 784)).astype("float32") / 255
    x_test = np.reshape(x_test, (10000, 784)).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    assert x_train.shape == (60000, 784)
    assert x_test.shape == (10000, 784)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    return (x_train, y_train), (x_test, y_test)

def main():
    model = Model()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        # recommended loss & metrics for MNIST, SparseCategoricalAccuracy is just
        # % of correct classifications, Adam with alpha=0.001 is default optimizer
    (x_train, y_train), (x_test, y_test) = load_mnist()
    # set up callbacks for TensorFlow Profiler
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir = 'logs',
      histogram_freq = 1,
      profile_batch = '0,600')
    # train model
    model.fit(x=x_train, y=y_train, batch_size=100, callbacks=[tensorboard_callback])
    # test model
    model.evaluate(x=x_test, y=y_test, batch_size=100)

if __name__ == '__main__':
    main()