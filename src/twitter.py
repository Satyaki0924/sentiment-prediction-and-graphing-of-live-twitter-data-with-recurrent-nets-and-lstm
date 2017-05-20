from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from src.preprocess import Process
from twitter_setup import Setup
from src.lstm import LSTM
import tensorflow as tf
import pickle
import numpy as np
import os

graph = tf.Graph()
path = os.path.dirname(os.path.realpath(__file__))
lstm_size, lstm_layers, batch_size, learning_rate = Setup().get_val()
with open(os.path.dirname(os.path.realpath(path)) + '/vocab_list/data_vocab.p', 'rb') as f:
    n_words = len(pickle.load(f))
_, _inputs_, _, keep_prob, _, _, _, _, _, saver, predictions = \
    LSTM(n_words, graph, lstm_size, lstm_layers, batch_size, learning_rate).get()
c_key, c_secret, acc_token, acc_secret = Setup().get()
kyword = str(input('Enter keyword (eg: Trump)\n>> '))

class Listener(StreamListener):
    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data["text"]
        print(tweet)
        features, _ = Process().process_reviews(tweet)
        for i in features:
            i = np.reshape(i, (1, len(i)))
            with tf.Session(graph=graph) as sess:
                saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
                feed = {_inputs_: i, keep_prob: 1}
                pred = sess.run([predictions], feed_dict=feed)
                print(pred[0][0][0])
                with open(os.path.dirname(os.path.realpath(path)) + 'points/'+kyword+'-plot.txt', 'a') as file:
                    file.write(str(pred[0][0][0]) + '\n')
        return True

    def on_error(self, status):
        print(status)


auth = OAuthHandler(c_key, c_secret)
auth.set_access_token(acc_token, acc_secret)
twitterStream = Stream(auth, Listener())
twitterStream.filter(track=[kyword])
