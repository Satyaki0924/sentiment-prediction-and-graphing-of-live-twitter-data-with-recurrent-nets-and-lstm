from __future__ import print_function
from src.preprocess import Process
from src.lstm import LSTM
import numpy as np
from src.split import Split
from time import time
import tensorflow as tf
import pickle
import os


class Main(object):
    def __init__(self, lstm_size, lstm_layers, batch_size, learning_rate):
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        with open(os.path.dirname(os.path.realpath(self.path)) + '/data/reviews.txt', 'r') as f:
            reviews = f.read()
        with open(os.path.dirname(os.path.realpath(self.path)) + '/data/labels.txt', 'r') as f:
            labels = f.read()
        self.features, non_zero_idx = Process().process_reviews(reviews)
        self.labels = Process().process_labels(labels, non_zero_idx)
        self.graph = tf.Graph()
        with open(os.path.dirname(os.path.realpath(self.path)) + '/vocab_list/data_vocab.p', 'rb') as f:
            self.n_words = len(pickle.load(f))

    def get_batches(self, x, y, batch_size=100):
        batches = len(x) // batch_size
        x, y = x[:batches * batch_size], y[:batches * batch_size]
        for i in range(0, len(x), batch_size):
            yield x[i:i + batch_size], y[i:i + batch_size]

    def train(self):
        print('Training dataset...')
        loss_pt = []
        acc_pt = []
        time_pt = []
        epochs = 10
        train_X, train_y = Split(self.features, self.labels).train_split()
        val_X, _, val_y, _ = Split(self.features, self.labels).test_split()
        initial_state, _inputs_, _labels_, keep_prob, cost, final_state, optimizer, cell, accuracy, saver, _ \
            = LSTM(self.n_words, self.graph, self.lstm_size, self.lstm_layers, self.batch_size,
                   self.learning_rate).get()
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 1
            for e in range(epochs):
                start = time()
                state = sess.run(initial_state)
                for ii, (x, y) in enumerate(self.get_batches(train_X, train_y, self.batch_size), 1):
                    feed = {_inputs_: x, _labels_: y[:, None], keep_prob: 0.5, initial_state: state}
                    loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
                    if iteration % 5 == 0:
                        print("Epoch: {}/{}".format(e, epochs),
                              "Iteration: {}".format(iteration),
                              "Train loss: {:.3f}".format(loss))
                        loss_pt.append(loss)
                    if iteration % 25 == 0:
                        val_acc = []
                        val_state = sess.run(cell.zero_state(self.batch_size, tf.float32))
                        for x, y in self.get_batches(val_X, val_y, self.batch_size):
                            feed = {_inputs_: x, _labels_: y[:, None], keep_prob: 1, initial_state: val_state}
                            batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                            val_acc.append(batch_acc)
                        print("Val acc: {:.3f}".format(np.mean(val_acc)))
                        acc_pt.append(val_acc)
                    stop = time()
                    time_pt.append(stop - start)
                    iteration += 1
            saver.save(sess, os.path.dirname(os.path.realpath(self.path)) + "/checkpoints/sentiment.ckpt")
            loss_pt = list(map(str, loss_pt))
            acc_pt = list(map(str, acc_pt))
            time_pt = list(map(str, time_pt))
            with open(os.path.dirname(os.path.realpath(self.path)) + 'points/loss.p', 'wb') as f:
                pickle.dump(list(loss_pt), f)
            with open(os.path.dirname(os.path.realpath(self.path)) + 'points/acc.p', 'wb') as f:
                pickle.dump(list(acc_pt), f)
            with open(os.path.dirname(os.path.realpath(self.path)) + 'points/time.b', 'wb') as f:
                pickle.dump(list(time_pt), f)

    def test_accuracy(self):
        print('Testing accuracy...')
        test_acc = []
        initial_state, _inputs_, _labels_, keep_prob, _, final_state, _, cell, accuracy, saver, _ \
            = LSTM(self.n_words, self.graph, self.lstm_size, self.lstm_layers, self.batch_size,
                   self.learning_rate).get()
        _, test_X, _, test_y = Split(self.features, self.labels).test_split()
        with tf.Session(graph=self.graph) as sess:
            saver.restore(sess,
                          tf.train.latest_checkpoint(os.path.dirname(os.path.realpath(self.path)) + '/checkpoints'))
            test_state = sess.run(cell.zero_state(self.batch_size, tf.float32))
            for ii, (x, y) in enumerate(self.get_batches(test_X, test_y, self.batch_size), 1):
                feed = {_inputs_: x,
                        _labels_: y[:, None],
                        keep_prob: 1,
                        initial_state: test_state}
                batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
                test_acc.append(batch_acc)
            print("Test accuracy: {:.3f}".format(np.mean(test_acc)))

    def manual_testing(self):
        _, _inputs_, _, keep_prob, _, _, _, _, _, saver, predictions \
            = LSTM(self.n_words, self.graph, self.lstm_size, self.lstm_layers, self.batch_size,
                   self.learning_rate).get()
        while True:
            try:
                sentence = str(input('Enter your sentence...\n'))
                features, _ = Process().process_reviews(sentence)
                f = np.zeros((int(500 - len(features)), 200), dtype=int)
                z = np.concatenate((features, f))
                with tf.Session(graph=self.graph) as sess:
                    saver.restore(sess, tf.train.latest_checkpoint(
                        os.path.dirname(os.path.realpath(self.path)) + '/checkpoints'))
                    feed = {_inputs_: z, keep_prob: 1}
                    pred = sess.run([predictions], feed_dict=feed)
                    if pred[0][0] < 0.5:
                        print('negative')
                    else:
                        print('positive')
                out = int(input('Enter 1 to quit and any key to continue...'))
                if out == 1:
                    break
            except:
                pass

    def execute(self, dec):
        print('*****Author: Satyaki Sanyal*****')
        print('***This project must only be used for educational purpose***')
        if dec == 1:
            Main(self.lstm_size, self.lstm_layers, self.batch_size, self.learning_rate).train()
        elif dec == 2:
            Main(self.lstm_size, self.lstm_layers, self.batch_size, self.learning_rate).test_accuracy()
        elif dec == 3:
            Main(self.lstm_size, self.lstm_layers, self.batch_size, self.learning_rate).manual_testing()
        else:
            raise Exception('InvalidArgumentError: Input not recognized.')
