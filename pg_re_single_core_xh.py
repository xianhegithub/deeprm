import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import scipy.signal
from tqdm import tqdm
import tensorflow as tf


import environment_xh
import slow_down_cdf_xh
import job_distribution_xh


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    assert x.ndim >= 1
    # More efficient version:
    out = scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out  # output np.array y[i]


def get_entropy(vec):
    entropy = - np.sum(vec * np.log(vec))
    if np.isnan(entropy):
        entropy = 0
    return entropy



def concatenate_all_ob(trajs, pa):

    timesteps_total = 0
    for i in range(len(trajs)):
        timesteps_total += len(trajs[i]['reward'])

    all_ob = np.zeros(
        (timesteps_total, 1, pa.network_input_height, pa.network_input_width),
        dtype=np.float32)

    timesteps = 0
    for i in range(len(trajs)):
        for j in range(len(trajs[i]['reward'])):
            all_ob[timesteps, 0, :, :] = trajs[i]['ob'][j]
            timesteps += 1

    return all_ob


def concatenate_all_ob_across_examples(all_ob, pa):
    num_epis = len(all_ob)
    total_samp = 0
    for i in range(num_epis):
        total_samp += all_ob[i].shape[0]

    all_ob_contact = np.zeros(
        (total_samp, 1, pa.network_input_height, pa.network_input_width),
        dtype=np.float32)

    total_samp = 0

    for i in range(num_epis):
        prev_samp = total_samp
        total_samp += all_ob[i].shape[0]
        all_ob_contact[prev_samp : total_samp, :, :, :] = all_ob[i]

    return all_ob_contact


def process_all_info(trajs):
    enter_time = []
    finish_time = []
    job_len = []

    for traj in trajs:
        enter_time.append(np.array([traj['info'].record[i].enter_time for i in range(len(traj['info'].record))]))
        finish_time.append(np.array([traj['info'].record[i].finish_time for i in range(len(traj['info'].record))]))
        job_len.append(np.array([traj['info'].record[i].len for i in range(len(traj['info'].record))]))

    enter_time = np.concatenate(enter_time)
    finish_time = np.concatenate(finish_time)
    job_len = np.concatenate(job_len)

    return enter_time, finish_time, job_len


def plot_lr_curve(output_file_prefix, max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                  ref_discount_rews, ref_slow_down):
    num_colors = len(ref_discount_rews) + 2
    cm = plt.get_cmap('gist_rainbow')

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121)
    col = [cm(1. * i / num_colors) for i in range(num_colors)]

    ax.plot(mean_rew_lr_curve, linewidth=2, label='PG mean')
    for plot_idx, k in enumerate(ref_discount_rews):
        ax.plot(np.tile(np.average(ref_discount_rews[k]), len(mean_rew_lr_curve)), linewidth=2, label=k, color = col[plot_idx])
    ax.plot(max_rew_lr_curve, linewidth=2, label='PG max')

    plt.legend(loc=4)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Discounted Total Reward", fontsize=20)

    ax = fig.add_subplot(122)
    col = [cm(1. * i / num_colors) for i in range(num_colors)]

    ax.plot(slow_down_lr_curve, linewidth=2, label='PG mean')
    for plot_idx, k in enumerate(ref_discount_rews):
        ax.plot(np.tile(np.average(np.concatenate(ref_slow_down[k])), len(slow_down_lr_curve)), linewidth=2, label=k, color = col[plot_idx])

    plt.legend(loc=1)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Slowdown", fontsize=20)

    plt.savefig(output_file_prefix + "_lr_curve" + ".pdf")


def launch(pa, pg_resume=None, render=False, repre='image', end='no_new_job'):
    nw_len_epis, nw_size_epis = job_distribution_xh.generate_episodes_work(pa)  # seed = 42

    env = environment_xh.Env(pa, nw_len_epis=nw_len_epis, nw_size_epis=nw_size_epis,
                             render=False, repre=repre, end=end)

    tf.reset_default_graph()

    global_step = tf.Variable(0, name="global_step", trainable=False)

    print('network_input_height=', pa.network_input_height)
    print('network_input_width=', pa.network_input_width)
    print('network_output_dim=', pa.network_output_dim)

    with tf.variable_scope('pg_graph'):
        net_states = tf.placeholder(shape=(None, 1, pa.network_input_height, pa.network_input_width), dtype=tf.float32,
                                     name="states")  # shape=(1, self.input_height, self.input_width),

        net_action = tf.placeholder(dtype=tf.int32, name="action")
        net_values = tf.placeholder(dtype=tf.float32, name="value")

        # ===================================
        # build neural network
        # ===================================
        # out = tf.reshape(self.states, shape=(-1, self.input_width * self.input_height))
        out = tf.keras.layers.Flatten()(net_states)
        # out = tf.contrib.layers.flatten(self.states)
        out = tf.keras.layers.Dense(units=20, activation=tf.nn.relu,
                                    kernel_initializer=tf.initializers.random_normal(stddev=0.01),
                                    bias_initializer=tf.zeros_initializer(),
                                    )(out)
        logits_action_probs = tf.keras.layers.Dense(units=pa.network_output_dim, activation=None,
                                                         kernel_initializer=tf.initializers.random_normal(stddev=0.01),
                                                         bias_initializer=tf.zeros_initializer(),
                                                         )(out)

        # ===================================
        # reinforcement learning loss and train op
        # ===================================
        net_action_probs = tf.squeeze(tf.nn.softmax(logits_action_probs))

        index = tf.transpose(tf.stack([tf.range(tf.shape(net_action_probs)[0]), net_action]))
        net_rl_picked_action_prob = tf.gather_nd(net_action_probs, index)
        net_rl_loss = tf.reduce_mean(-tf.log(net_rl_picked_action_prob) * net_values)
        net_rl_optimizer = tf.train.AdamOptimizer(learning_rate=pa.lr_rate)
        #net_rl_optimizer = tf.train.RMSPropOptimizer(learning_rate=pa.lr_rate, epsilon=1e-8)
        net_rl_train_op = net_rl_optimizer.minimize(net_rl_loss, global_step=tf.train.get_global_step())


    # ----------------------------
    print("Preparing for data...")
    # ----------------------------

    ref_discount_rews, ref_slow_down = slow_down_cdf_xh.launch(pa, pg_resume=None, render=False, plot=False,
                                                               repre=repre, end=end)

    mean_rew_lr_curve = []
    max_rew_lr_curve = []
    slow_down_lr_curve = []

    timer_start = time.time()
    np.random.seed(20)
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('/home/xyangzha/Documents/deeprm-master/deeprm-master/rl_train')
        for iteration in tqdm(range(pa.num_epochs)):
            summary = tf.Summary()
            all_ob = []
            all_action = []
            all_adv = []
            all_eprews = []
            all_eplens = []
            all_slowdown = []
            all_entropy = []

            # go through all examples
            for _ in tqdm(range(pa.num_epis)):

                # Collect trajectories until we get timesteps_per_batch total timesteps
                trajs = []

                #======generate trajectory by simulation with current policy
                for _ in range(pa.num_epis_per_batch):
                    env.reset()
                    obs = []
                    acts = []
                    rews = []
                    entropy = []
                    info = []

                    ob = env.observe()

                    for _ in range(pa.episode_max_length):
                        state = np.zeros([1, 1, 20, 224])
                        state[0, 0, :, :] = ob
                        act_prob = sess.run(net_action_probs, {net_states: state})
                        a = np.random.choice(len(np.asarray(act_prob)), p=np.asarray(act_prob))

                        obs.append(ob)  # store the ob at current decision making step
                        acts.append(a)

                        ob, rew, done, info = env.step(a, repeat=True)

                        rews.append(rew)
                        entropy.append(get_entropy(act_prob))

                        if done: break
                        if render: env.render()

                    traj = {'reward': np.array(rews),
                            'ob': np.array(obs),
                            'action': np.array(acts),
                            'entropy': entropy,
                            'info': info
                            }

                    trajs.append(traj)
                #=================================================

                # roll to next example
                env.epi_idx = (env.epi_idx + 1) % env.pa.num_epis

                all_ob.append(concatenate_all_ob(trajs, pa))

                # Compute discounted sums of rewards
                rets = [discount(traj["reward"], pa.discount) for traj in trajs]
                maxlen = max(len(ret) for ret in rets)
                padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

                # Compute time-dependent baseline
                baseline = np.mean(padded_rets, axis=0)

                # Compute advantage function
                advs = [ret - baseline[:len(ret)] for ret in rets]
                all_action.append(np.concatenate([traj["action"] for traj in trajs]))
                all_adv.append(np.concatenate(advs))

                all_eprews.append(
                    np.array([discount(traj["reward"], pa.discount)[0] for traj in trajs]))  # episode total rewards
                all_eplens.append(np.array([len(traj["reward"]) for traj in trajs]))  # episode lengths

                # All Job Stat
                enter_time, finish_time, job_len = process_all_info(trajs)
                finished_idx = (finish_time >= 0)
                all_slowdown.append(
                    (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]
                )

                # Action prob entropy
                all_entropy.append(np.concatenate([traj["entropy"]]))

            all_ob = concatenate_all_ob_across_examples(all_ob, pa)
            all_action = np.concatenate(all_action)
            all_adv = np.concatenate(all_adv)

            # Do policy gradient update step
            feed_dict = {net_states: all_ob, net_action: all_action, net_values: all_adv}
            _, loss = sess.run([net_rl_train_op, net_rl_loss], feed_dict)

            eprews = np.concatenate(all_eprews)  # episode total rewards
            eplens = np.concatenate(all_eplens)  # episode lengths

            all_slowdown = np.concatenate(all_slowdown)

            all_entropy = np.concatenate(all_entropy)

            timer_end = time.time()

            print("-----------------")
            print("Iteration: \t %i" % iteration)
            print("NumTrajs: \t %i" % len(eprews))
            print("NumTimesteps: \t %i" % np.sum(eplens))
            print("Loss:     \t %s" % loss)
            print("MaxRew: \t %s" % np.average([np.max(rew) for rew in all_eprews]))
            print("MeanRew: \t %s +- %s" % (eprews.mean(), eprews.std()))
            print("MeanSlowdown: \t %s" % np.mean(all_slowdown))
            print("MeanLen: \t %s +- %s" % (eplens.mean(), eplens.std()))
            print("MeanEntropy \t %s" % (np.mean(all_entropy)))
            print("Elapsed time\t %s" % (timer_end - timer_start), "seconds")
            print("-----------------")
            summary.value.add(tag="loss", simple_value=loss)
            summary.value.add(tag="MeanReward", simple_value=eprews.mean())
            summary.value.add(tag="MeanSlowdown", simple_value=np.mean(all_slowdown))
            writer.add_summary(summary, iteration)

            timer_start = time.time()

            max_rew_lr_curve.append(np.average([np.max(rew) for rew in all_eprews]))
            mean_rew_lr_curve.append(eprews.mean())
            slow_down_lr_curve.append(np.mean(all_slowdown))

            if iteration % pa.output_freq == 0:
                #param_file = open(pa.output_filename + '_' + str(iteration) + '.pkl', 'wb')
                #pickle.dump(pg_learner.get_params(), param_file, -1)
                #param_file.close()

                # slow_down_cdf.launch(pa, pa.output_filename + '_' + str(iteration) + '.pkl',
                # render=False, plot=True, repre=repre, end=end)

                plot_lr_curve(pa.output_filename,
                              max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                              ref_discount_rews, ref_slow_down)


def main():
    import parameters_xh

    pa = parameters_xh.Parameters()

    pa.num_steps_in_epi = 50
    pa.num_epis = 10

    pa.compute_dependent_parameters()

    pg_resume = None
    # pg_resume = 'data/pg_su_net_file_450.pkl'

    render = False

    launch(pa, pg_resume, render, repre='image', end='all_done')


if __name__ == '__main__':
    main()
