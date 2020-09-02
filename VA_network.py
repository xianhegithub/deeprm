import tensorflow as tf



class ValueEstimator:
    def __init__(self, pa, scope="va_graph"):

        self.input_height = pa.network_input_height
        self.input_width = pa.network_input_width
        self.output_height = pa.network_output_dim
        self.lr_rate = pa.lr_rate


        print ('network_input_height=', pa.network_input_height)
        print ('network_input_width=', pa.network_input_width)
        print ('network_output_dim=', pa.network_output_dim)

        self.num_frames = pa.num_frames

        self.update_counter = 0
        with tf.variable_scope(scope):
            self.states = tf.placeholder(shape=(None, 1, self.input_height, self.input_width), dtype=tf.float32, name="states")
            self.values = tf.placeholder(dtype=tf.float32, name="values")

            # ===================================
            # build neural network
            # ===================================
            regularizer_scale = 1e-3
            out = tf.reshape(self.states, [-1, self.input_height * self.input_width])
            out = tf.layers.Dense(units=20, activation=tf.nn.relu,
                                  kernel_initializer=tf.initializers.random_normal(stddev=0.01),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=regularizer_scale),
                                  bias_initializer=tf.zeros_initializer()
                                  )(out)
            self.logits_action_probs = tf.layers.Dense(units=self.output_height, activation=None,
                                                      kernel_initializer=tf.initializers.random_normal(stddev=0.01),
                                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=regularizer_scale),
                                                      bias_initializer=tf.zeros_initializer()
                                                      )(out)

            # ==================================
            # supervised learning Loss and train op
            #===================================

            self.su_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=self.logits_action_probs, onehot_labels=self.targets))
            regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope)
            self.su_loss += tf.reduce_sum(regu_losses)

            self.su_optimizer = tf.train.AdamOptimizer(learning_rate=pa.lr_rate)
            self.su_train_op = self.su_optimizer.minimize(self.su_loss, global_step=tf.train.get_global_step())
            correct_prediction = tf.equal(tf.argmax(self.logits_action_probs, axis=1, output_type=tf.int32), tf.argmax(self.targets, axis=1, output_type=tf.int32))
            self.su_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def su_test(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.states: state, self.targets: target}
        loss, acc = sess.run([self.su_loss, self.su_accuracy], feed_dict)
        return loss, acc

    def su_train(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.states: state, self.targets: target}
        _, loss, acc = sess.run([self.su_train_op, self.su_loss, self.su_accuracy], feed_dict)
        return loss, acc

    def predict(self, input4pred, sess=None):
        sess = sess or tf.get_default_session()
        action_probs, random_action = sess.run([self.action_probs, self.random_action], {self.states: input4pred})
        return action_probs, random_action




