import tensorflow as tf


class LSTM(object):
    def __init__(self, n_words, graph, lstm_size, lstm_layers, batch_size, learning_rate):
        self.n_words = n_words
        self.graph = graph
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def define_param(self):
        with self.graph.as_default():
            self._inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
            self._labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def embedding(self):
        embed_size = 300
        with self.graph.as_default():
            embedding = tf.Variable(tf.random_uniform((self.n_words, embed_size), -1, 1), name='W')
            self.embed = tf.nn.embedding_lookup(embedding, self._inputs_)

    def lstm(self):
        with self.graph.as_default():
            self.lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size, reuse=tf.get_variable_scope().reuse)
            self.drop = tf.contrib.rnn.DropoutWrapper(self.lstm, output_keep_prob=self.keep_prob)
            self.cell = tf.contrib.rnn.MultiRNNCell([self.drop] * self.lstm_layers)
            self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

    def rnn(self):
        with self.graph.as_default():
            self.outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, self.embed, initial_state=self.initial_state)

    def output(self):
        with self.graph.as_default():
            self.predictions = tf.contrib.layers.fully_connected(self.outputs[:, -1], 1, activation_fn=tf.sigmoid)
            self.cost = tf.losses.mean_squared_error(self._labels_, self.predictions)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def validation_accuracy(self):
        with self.graph.as_default():
            self.correct_pred = tf.equal(tf.cast(tf.round(self.predictions), tf.int32), self._labels_)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def def_saver(self):
        with self.graph.as_default():
            self.saver = tf.train.Saver()

    def get(self):
        print('setting up lstm and dynamic recurrent neural network... ***Use GPU***')
        self.define_param()
        self.embedding()
        self.lstm()
        self.rnn()
        self.output()
        self.validation_accuracy()
        self.def_saver()
        return self.initial_state, self._inputs_, \
               self._labels_, self.keep_prob, self.cost, \
               self.final_state, self.optimizer, self.cell, self.accuracy, self.saver, self.predictions

