import tensorflow as tf



class PGLearner:
    def __init__(self, pa, scope="pg_graph"):

        self.input_height = pa.network_input_height
        self.input_width = pa.network_input_width
        self.output_height = pa.network_output_dim



        print ('network_input_height=', pa.network_input_height)
        print ('network_input_width=', pa.network_input_width)
        print ('network_output_dim=', pa.network_output_dim)

        self.num_frames = pa.num_frames

        self.update_counter = 0
        with tf.variable_scope(scope):
            self.states = tf.placeholder(shape=(None, 1, self.input_height, self.input_width), dtype=tf.float32, name="states")#shape=(1, self.input_height, self.input_width),
            self.targets = tf.placeholder(dtype=tf.int32, name="targets")

            self.action = tf.placeholder(dtype=tf.int32, name="action")
            self.values = tf.placeholder(dtype=tf.float32, name="value")

            # ===================================
            # build neural network
            # ===================================
            regularizer_scale = 1e-3
            #out = tf.reshape(self.states, [-1, self.input_height * self.input_width])

            #self.states = tf.keras.layers.Input(shape=(1, self.input_height, self.input_width), dtype=tf.float32, name="states")
            out = tf.keras.layers.Flatten()(self.states)
            out =tf.keras.layers.Dense(units=20, activation=tf.nn.relu,
                                    kernel_initializer=tf.initializers.random_normal(stddev=0.01),
                                    bias_initializer=tf.zeros_initializer(),
                                    )(out)

            self.logits_action_probs = tf.keras.layers.Dense(units=pa.network_output_dim, activation=None,
                                                         kernel_initializer=tf.initializers.random_normal(stddev=0.01),
                                                         bias_initializer=tf.zeros_initializer(),
                                                         )(out)


            # ==================================
            # supervised learning Loss and train op
            #===================================

            self.su_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=self.logits_action_probs, onehot_labels=self.targets))
            self.su_loss += tf.losses.get_regularization_loss()#tf.reduce_sum(reg_losses)

            self.su_optimizer = tf.train.RMSPropOptimizer(learning_rate=pa.lr_rate, decay=pa.rms_rho, epsilon=pa.rms_eps)
            self.su_train_op = self.su_optimizer.minimize(self.su_loss, global_step=tf.train.get_global_step())
            correct_prediction = tf.equal(tf.argmax(self.logits_action_probs, axis=1, output_type=tf.int32), tf.argmax(self.targets, axis=1, output_type=tf.int32))
            self.su_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


            #===================================
            # reinforcement learning loss and train op
            #===================================
            self.action_probs = tf.squeeze(tf.nn.softmax(self.logits_action_probs))
            self.index = tf.transpose(tf.stack([tf.range(tf.shape(self.action_probs)[0]), self.action]))
            self.rl_picked_action_prob = tf.gather_nd(self.action_probs, self.index)
            self.rl_loss = tf.reduce_mean(-tf.log(self.rl_picked_action_prob) * self.values)
            #self.rl_optimizer = tf.train.RMSPropOptimizer(learning_rate=pa.lr_rate, decay=pa.rms_rho, epsilon=pa.rms_eps)
            self.rl_optimizer = tf.train.AdamOptimizer(learning_rate=pa.lr_rate)
            self.rl_train_op = self.rl_optimizer.minimize(self.rl_loss, global_step=tf.train.get_global_step())

    def su_test(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.states: state, self.targets: target}
        loss, acc = sess.run([self.su_loss, self.su_accuracy], feed_dict)
        return loss, acc

    def su_train(self, state, target,  sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.states: state, self.targets: target}
        _, loss, acc = sess.run([self.su_train_op, self.su_loss, self.su_accuracy], feed_dict)#, options=options, run_metadata=run_metadata)
        return loss, acc

    def predict(self, input4pred, sess=None):
        sess = sess or tf.get_default_session()
        action_probs = sess.run(self.action_probs, {self.states: input4pred})
        return action_probs

    def rl_train(self, state, action, value, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.states: state, self.action: action, self.values: value}
        _, loss = sess.run([self.rl_train_op, self.rl_loss], feed_dict)
        return loss

    def debug(self, state, action, value, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.states: state, self.action: action, self.values: value}
        action_probs, picked_action_prob, action, value = sess.run([self.action_probs, self.rl_picked_action_prob, self.action, self.values], feed_dict)
        return action_probs, picked_action_prob, action, value


