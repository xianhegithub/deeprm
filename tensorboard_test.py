import numpy as np
import time
import sys
import pickle
import tensorflow as tf

import environment_xh
import pg_network_su_xh
import other_policies_xh
import job_distribution_xh

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

def add_sample(x, y, idx, x_to_add, y_to_add):
    x[idx, 0, :, :] = x_to_add
    y[idx] = y_to_add


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def one_hot_encoding(input_array):
    max_value = np.max(input_array)+1
    return np.eye(max_value)[input_array]


def prepare_training_data(pa, current_policy, env):
    mem_alloc = 4

    x = np.zeros([pa.num_steps_in_epi * pa.num_epis * mem_alloc, 1,
                  pa.network_input_height, pa.network_input_width],
                 dtype=np.float32)
    y = np.zeros(pa.num_steps_in_epi * pa.num_epis * mem_alloc,
                 dtype='int32')

    counter = 0

    data_exists = os.path.isfile('episode_history.pkl')

    if data_exists:
        pkl_file = open('episode_history.pkl', 'rb')
        episode_data = pickle.load(pkl_file)
        x_train = episode_data[0]
        x_test = episode_data[1]
        y_train = episode_data[2]
        y_test = episode_data[3]
        print('Loaded training data')
    else:
        for train_epi_idx in range(pa.num_epis):

            env.reset()

            for _ in range(pa.episode_max_length):

                # ---- get current state ----
                ob = env.observe()

                a = current_policy(env.machine, env.job_slot)

                if counter < pa.num_steps_in_epi * pa.num_epis * mem_alloc:
                    add_sample(x, y, counter, ob,
                               a)  # add X and y samples to the list, X is the obversation (state), y is the action index.
                    counter += 1

                ob, rew, done, info = env.step(a, repeat=True)

                if done:  # hit void action, exit
                    break

            # roll to next example
            env.epi_idx = (env.epi_idx + 1) % env.pa.num_epis

        num_train = int(0.8 * counter)
        num_test = int(0.2 * counter)
        y = one_hot_encoding(y)

        x_train, x_test = x[:num_train], x[num_train: num_train + num_test]
        y_train, y_test = y[:num_train], y[num_train: num_train + num_test]

        output = open('episode_history.pkl', 'wb')
        pickle.dump([x_train, x_test, y_train, y_test], output)
        output.close()
        print('Locally generated data and saved')

    return x_train, x_test, y_train, y_test

def launch(pa, pg_resume=None, render=False, repre='image', end='no_new_job'):

    pa.unseen = False
    nw_len_epis, nw_size_epis = job_distribution_xh.generate_episodes_work(pa)
    env = environment_xh.Env(pa, nw_len_epis=nw_len_epis, nw_size_epis=nw_size_epis, render=False, repre='image', end=end)

    tf.reset_default_graph()
    global_step = tf.Variable(0, name="global_step", trainable=False)
    pg_learner = pg_network_su_xh.PGLearner(pa)


    if pa.evaluate_policy_name == "SJF":
        evaluate_policy = other_policies_xh.get_sjf_action
    elif pa.evaluate_policy_name == "PACKER":
        evaluate_policy = other_policies_xh.get_packer_action
    else:
        print("Panic: no policy known to evaluate.")
        exit(1)

    # ----------------------------
    print("Preparing for data...")
    # ----------------------------
    # --------generate or load many episodes for training--------------------------------------
    X_train, X_test, y_train, y_test = prepare_training_data(pa, evaluate_policy, env)

    # ----------------------------
    print("Start training...")
    # ----------------------------
    saver = tf.train.Saver()
    tf.random.set_random_seed(1234)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #writer = tf.summary.FileWriter('./graphs', sess.graph)
        if pg_resume is not None:
            saver.restore(sess, pg_resume)
            print("Model restored.")

        for epoch in range(pa.num_epochs):

            train_err = 0
            train_acc = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train, pa.batch_size, shuffle=True):
                inputs, targets = batch
                training_loss, training_acc = pg_learner.su_train(inputs, targets)
                train_err += training_loss
                train_acc += training_acc
                train_batches += 1


            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, y_test, pa.batch_size, shuffle=False):
                inputs, targets = batch
                testing_loss, testing_acc = pg_learner.su_test(inputs, targets)
                test_err += testing_loss
                test_acc += testing_acc
                test_batches += 1


        # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, pa.num_epochs, time.time() - start_time))
            print("  training loss:    \t\t{:.6f}".format(train_err / train_batches))
            print("  training accuracy:\t\t{:.2f} %".format(
                train_acc / float(train_batches) * 100))
            print("  test loss:        \t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:    \t\t{:.2f} %".format(
                test_acc / float(test_batches) * 100))
            sys.stdout.flush()
            if epoch % pa.output_freq == 0:
                save_path = saver.save(sess, "/home/xyangzha/Documents/deeprm-master/deeprm-master/SavedModel/model.ckpt", global_step=epoch)

        print("done")


def main():

    import parameters_xh

    pa = parameters_xh.Parameters()
    pa.num_epochs = 500
    pa.num_steps_in_epi = 1000  # 1000
    pa.num_epis = 50  # 100
    pa.num_nw = 10
    pa.num_epis_per_batch = 20
    pa.output_freq = 50
    # pa.max_nw_size = 5
    # pa.job_len = 5
    pa.new_job_rate = 0.3
    pa.episode_max_length = 10000  # 2000
    pa.compute_dependent_parameters()

    pg_resume = None
    #pg_resume = "/home/xyangzha/Documents/deeprm-master/deeprm-master/SavedModel/model.ckpt-5"

    render = False

    launch(pa, pg_resume, render, repre='image', end='all_done')


if __name__ == '__main__':
    main()
