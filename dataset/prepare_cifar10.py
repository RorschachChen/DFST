from tensorflow.keras.datasets import cifar10
import pickle

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
cifar_train = {}
cifar_test = {}

cifar_train['x_train'] = x_train
cifar_train['y_train'] = y_train

cifar_test['x_test'] = x_test
cifar_test['y_test'] = y_test

pickle.dump(cifar_train, open('./cifar_train', 'wb'))
pickle.dump(cifar_test, open('./cifar_test', 'wb'))