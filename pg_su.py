import numpy as np
import time
import sys
import pickle as cPickle
import tensorflow as tf

import environment_xh
import pg_network
import other_policies_xh
import job_distribution_xh

#np.set_printoptions(threshold='nan')


def add_sample(X, y, idx, X_to_add, y_to_add):
    X[idx, 0, :, :] = X_to_add
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


def launch(pa, pg_resume=None, render=False, repre='image', end='no_new_job'):

    pa.unseen = False
    nw_len_epis, nw_size_epis = job_distribution_xh.generate_episodes_work(pa)
    env = environment_xh.Env(pa, nw_len_epis=nw_len_epis, nw_size_epis=nw_size_epis, render=False, repre='image', end=end)

    pg_learner = pg_network.PGLearner(pa)

    if pg_resume is not None:
        net_handle = open(pg_resume, 'r')
        net_params = cPickle.load(net_handle)
        pg_learner.set_net_params(net_params)

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
    # generate many episodes for training

#    nw_len_seqs, nw_size_seqs = job_distribution_xh.generate_sequence_work(pa, seed=42)

    # print 'nw_time_seqs=', nw_len_seqs
    # print 'nw_size_seqs=', nw_size_seqs

    mem_alloc = 4

    X = np.zeros([pa.num_steps_in_epi * pa.num_epis * mem_alloc, 1,
                  pa.network_input_height, pa.network_input_width],
                 dtype=np.float32)
    y = np.zeros(pa.num_steps_in_epi * pa.num_epis * mem_alloc,
                 dtype='int32')

    print ('network_input_height=', pa.network_input_height)
    print ('network_input_width=', pa.network_input_width)

    counter = 0
#=======================Loop to genearate episodes according to evaluate_policy====================
    for train_epi_idx in range(pa.num_epis):

        env.reset()

        for _ in range(pa.episode_max_length):

            # ---- get current state ----
            ob = env.observe()

            a = evaluate_policy(env.machine, env.job_slot)

            if counter < pa.num_steps_in_epi * pa.num_epis * mem_alloc:

                add_sample(X, y, counter, ob, a)  # add X and y samples to the list, X is the obversation (state), y is the action index.
                counter += 1

            ob, rew, done, info = env.step(a, repeat=True)

            if done:  # hit void action, exit
                break

        # roll to next example
        env.epi_idx = (env.epi_idx + 1) % env.pa.num_epis

#===================================================================================================

    num_train = int(0.8 * counter)
    num_test = int(0.2 * counter)

    X_train, X_test = X[:num_train], X[num_train: num_train + num_test]
    y_train, y_test = y[:num_train], y[num_train: num_train + num_test]


    # ----------------------------
    print("Start training...")
    # ----------------------------
    for epoch in range(pa.num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, pa.batch_size, shuffle=True):
            inputs, targets = batch
            err, prob_act = pg_learner.su_train(inputs, targets)
            pg_act = np.argmax(prob_act, axis=1)
            train_err += err
            train_acc += np.sum(pg_act == targets)
            train_batches += 1

        # # And a full pass over the test data:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, pa.batch_size, shuffle=False):
            inputs, targets = batch
            err, prob_act = pg_learner.su_test(inputs, targets)
            pg_act = np.argmax(prob_act, axis=1)
            test_err += err
            test_acc += np.sum(pg_act == targets)
            test_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, pa.num_epochs, time.time() - start_time))
        print("  training loss:    \t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t\t{:.2f} %".format(
            train_acc / float(num_train) * 100))
        print("  test loss:        \t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:    \t\t{:.2f} %".format(
            test_acc / float(num_test) * 100))

        sys.stdout.flush()

        if epoch % pa.output_freq == 0:
            # this is where the code save the neural network parameters
            net_file = open(pa.output_filename + '_net_file_' + str(epoch) + '.pkl', 'wb')
            cPickle.dump(pg_learner.return_net_params(), net_file, -1)
            net_file.close()

    print("done")


def main():

    import parameters_xh

    pa = parameters_xh.Parameters()

    pa.num_steps_in_epi = 50
    pa.num_epis = 10


    pa.compute_dependent_parameters()

    pg_resume = None
    # pg_resume = 'data/tmp_450.pkl'

    render = False

    launch(pa, pg_resume, render, repre='image', end='all_done')


if __name__ == '__main__':
    main()
