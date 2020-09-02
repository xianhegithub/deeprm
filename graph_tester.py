import tensorflow as tf
import pickle as cPickle
import numpy as np
from tqdm import tqdm
class TestingGraph():
    def __init__(self, height, width, learning_rate=0.001, scope="TestingGraph"):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=(None, 1, height, width), dtype = tf.float32, name="inputs")
            self.targets = tf.placeholder(dtype=tf.float32, name="targets")

            regulariser_scale = 1e-2
            out = tf.reshape(self.inputs, [-1, height * width])
            out = tf.contrib.layers.fully_connected(out,
                                                    num_outputs=20,
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.initializers.random_normal(stddev=0.01),
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                        scale=regulariser_scale),
                                                    biases_initializer=tf.zeros_initializer()
                                                    )
            self.l_out_action_probs = tf.contrib.layers.fully_connected(out,
                                                    num_outputs=11,
                                                    activation_fn=None,
                                                    weights_initializer=tf.initializers.random_normal(stddev=0.01),
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                        scale=regulariser_scale),
                                                    biases_initializer=tf.zeros_initializer()
                                                    )
            # Loss and train op
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.l_out_action_probs, labels=self.targets))
            regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss += tf.reduce_sum(regu_losses)

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9, epsilon=1e-9)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())
            self.correct_prediction = tf.equal(tf.argmax(self.l_out_action_probs, axis=1, output_type=tf.int32), tf.argmax(self.targets, axis=1, output_type=tf.int32))
            self.su_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def su_test(self, input, target, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.su_accuracy, {self.inputs: input, self.targets: target})

    def su_train(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.inputs: state, self.targets: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


def main():
    pkl_file = open('episode_history.pkl', 'rb')
    episode_data = cPickle.load(pkl_file)
    X_train = episode_data[0]
    height = np.shape(np.asarray(X_train))[2]
    width = np.shape(np.asarray(X_train))[3]
    X_test = episode_data[1]
    y_train = episode_data[2]
    y_test = episode_data[3]

    tf.reset_default_graph()
    tf.random.set_random_seed(1234)
    global_step = tf.Variable(0, name="global_step", trainable=False)

    test_graph = TestingGraph(height,width)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i_episode in tqdm(range(100)):
            for _ in range(10):
                loss = test_graph.su_train(X_train, y_train)
                acc = test_graph.su_test(X_test, y_test)
                print(loss, acc)

if __name__ == '__main__':
    main()