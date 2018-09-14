import tensorflow as tf
import numpy as np

class SimpleNet:
    def __init__(self, n_user, n_movie, user_embed_size = 100, movie_embed_size = 300, keep_prob = 0.9):
        self._n_user = n_user
        self._n_movie = n_movie
        self._user_embed_size = user_embed_size
        self._movie_embed_size = movie_embed_size
        self._keep_prob = keep_prob

        with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
            self.create_network()


    def create_network(self):
        self._user = tf.placeholder(shape = [None], dtype = tf.int32)
        self._movie = tf.placeholder(shape = [None], dtype = tf.int32)
        self._batch_size = tf.placeholder(shape = [], dtype = tf.int32)
        self._keep_prob_tensor = tf.placeholder(shape = [], dtype = tf.float32)
        self._is_training = tf.placeholder(tf.bool)



        # Embedding layer:
        np.random.seed(0)
        user_embedding = tf.Variable(
                initial_value = tf.truncated_normal(
                    shape = [self._n_user, self._user_embed_size],
                    mean = 0.0,
                    stddev = 1.0),
                name="embedding"
        )
        self._user_embed = tf.nn.embedding_lookup(user_embedding, self._user, name = "user_embed")

        np.random.seed(0)
        movie_embedding = tf.Variable(
                initial_value = tf.truncated_normal(
                    shape = [self._n_movie, self._movie_embed_size],
                    mean = 0.0,
                    stddev = 1.0),
                name="embedding"
        )
        self._movie_embed = tf.nn.embedding_lookup(movie_embedding, self._movie, name = "movie_embed")

        # Shape: batch_size x (user_embed_size + movie_embed_size)
        self._X = tf.concat([self._user_embed, self._movie_embed], axis = -1)
        self._fc1 = self.feed_forward(
            self._X,
            name = "fc1",
            inp_channel = self._user_embed_size + self._movie_embed_size,
            op_channel = 100,
        )
        self._fc1_dropout = tf.nn.dropout(self._fc1, keep_prob = self._keep_prob_tensor)
        self._fc2 = self.feed_forward(self._fc1_dropout, inp_channel = 100, op_channel = 1, name = "fc2", op_layer = True)
        self._op = tf.clip_by_value(
            self._fc2,
            clip_value_max = 1.0,
            clip_value_min = 0.0
        )

        self._y = tf.placeholder(shape = [None], dtype = tf.float32)
        self._mean_loss = tf.reduce_mean(tf.square(self._op - self._y))

        self._sess = tf.Session()
        self._optimizer = tf.train.AdamOptimizer()
        self._train_step = self._optimizer.minimize(self._mean_loss)
        self._init_op = tf.global_variables_initializer()
        self._saver = tf.train.Saver()

    def fit(
            self, user, movie, y, user_val = None, movie_val = None, y_val = None,
            n_epoch = 100, batch_size = 16, patience = 5,
            weight_load_path = None, weight_save_path = None, print_every = 100
    ):
        if weight_load_path is not None:
            self.load_weight(weight_load_path)
        else:
            self._sess.run(self._init_op)

        indicies = np.arange(user.shape[0])
        iter_cnt = 0
        val_losses = []
        early_stopping_cnt = 0

        for e in range(n_epoch):
            print("Epoch " + str(e + 1))
            np.random.shuffle(indicies)

            n_batch = user.shape[0] // batch_size

            for i in range(n_batch):
                idx = indicies[i * batch_size:(i + 1)*batch_size]
                feed_dict = {
                    self._user: user[idx],
                    self._movie: movie[idx],
                    self._y: y[idx],
                    self._is_training: True,
                    self._keep_prob_tensor: self._keep_prob
                }

                loss, op, _ = self._sess.run([self._mean_loss, self._op, self._train_step], feed_dict = feed_dict)

                # if np.isnan(loss):
                #     if np.any(np.isnan(op)):
                #         print("both fucked")
                #     else:
                #         print(i)
                #         print(n_batch)
                #         print(user.shape[0])
                #         print(indicies[i * batch_size:(i + 1)*batch_size])
                #         print(op)
                #         print(y[idx])
                #         print(user[idx])
                #         print(movie[idx])
                #         print(np.mean(np.square(op - y[idx])))
                #         print(np.any(np.isnan(y[idx])))
                #         print("loss fuck first")
                #     return

                if iter_cnt % print_every == 0:
                    print("Iteration " + str(iter_cnt) + " with loss " + str(loss))

                iter_cnt += 1

            if user_val is not None:
                feed_dict = {
                    self._user: user_val,
                    self._movie: movie_val,
                    self._y: y_val,
                    self._is_training: False,
                    self._keep_prob_tensor: 1.0
                }
                val_loss = self._sess.run(self._mean_loss, feed_dict = feed_dict)
                print("Validation loss " + str(val_loss))
                val_losses.append(val_loss)


                if val_loss == np.min(val_losses):
                    early_stopping_cnt = 0
                    if weight_save_path is not None:
                        print("Model improves.")
                        self.save_weight(weight_save_path)
                        print("Model saved at " + weight_save_path)
                else:
                    early_stopping_cnt += 1
                    if early_stopping_cnt > patience:
                        print("Patience exceeded. Finish training")
                        return


    def predict(self, user, movie):
        feed_dict = {
            self._user: user,
            self._movie: movie,
            self._is_training: False,
            self._keep_prob_tensor: 1.0
        }
        return self._sess.run(self._op, feed_dict = feed_dict).astype(np.uint8)

    def save_weight(self, weight_save_path):
        self._saver.save(
            sess = self._sess,
            save_path = weight_save_path
        )

    def load_weight(self, weight_load_path):
        self._sess.run(self._init_op)
        self._saver.restore(
            sess = self._sess,
            save_path = weight_load_path
        )

    def feed_forward(self, x, name, inp_channel, op_channel, op_layer=False, norm = True):
        W = tf.get_variable("W_" + name, shape=[inp_channel, op_channel], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b_" + name, shape=[op_channel], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        z = tf.matmul(x, W) + b
        if op_layer:
            # a = tf.nn.sigmoid(z)
            # return a
            return z
        else:
            a = tf.nn.relu(z)
            if norm:
                a_norm = tf.layers.batch_normalization(a, training=self._is_training)
                return a_norm

            return a

    def evaluate(self, user, movie, y):
        feed_dict = {
            self._user: user,
            self._movie: movie,
            self._y: y,
            self._is_training: False
        }
        return self._sess.run(self._mean_loss, feed_dict=feed_dict)




