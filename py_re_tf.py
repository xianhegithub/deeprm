import numpy as np
import time
import pickle as cPickle
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm

import environment_xh
import pg_network
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


def get_traj(agent, env, episode_max_length, render=False):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """
    env.reset()
    obs = []
    acts = []
    rews = []
    entropy = []
    info = []

    ob = env.observe()

    for _ in range(episode_max_length):
        act_prob = agent.get_one_act_prob(ob)
        a = np.random.choice(len(np.asarray(act_prob)), p = np.asarray(act_prob))

        obs.append(ob)  # store the ob at current decision making step
        acts.append(a)

        ob, rew, done, info = env.step(a, repeat=True)

        rews.append(rew)
        entropy.append(get_entropy(act_prob))

        if done: break
        if render: env.render()

    return {'reward': np.array(rews),
            'ob': np.array(obs),
            'action': np.array(acts),
            'entropy': entropy,
            'info': info
            }


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

    pg_learner = pg_network.PGLearner(pa)

    if pg_resume is not None:
        net_handle = open(pg_resume, 'rb')
        net_params = cPickle.load(net_handle)
        pg_learner.set_net_params(net_params)

    # ----------------------------
    print("Preparing for data...")
    # ----------------------------

    ref_discount_rews, ref_slow_down = slow_down_cdf_xh.launch(pa, pg_resume=None, render=False, plot=False, repre=repre, end=end)

    mean_rew_lr_curve = []
    max_rew_lr_curve = []
    slow_down_lr_curve = []

    timer_start = time.time()

    for iteration in tqdm(range(pa.num_epochs)):

        all_ob = []
        all_action = []
        all_adv = []
        all_eprews = []
        all_eplens = []
        all_slowdown = []
        all_entropy = []

        # go through all examples
        for loop_epi_idx in tqdm(range(pa.num_epis)):

            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajs = []

            for _ in range(pa.num_epis_per_batch):
                traj = get_traj(pg_learner, env, pa.episode_max_length)
                trajs.append(traj)

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

            all_eprews.append(np.array([discount(traj["reward"], pa.discount)[0] for traj in trajs]))  # episode total rewards
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
        loss = pg_learner.train(all_ob, all_action, all_adv)
        eprews = np.concatenate(all_eprews)  # episode total rewards
        eplens = np.concatenate(all_eplens)  # episode lengths

        all_slowdown = np.concatenate(all_slowdown)

        all_entropy = np.concatenate(all_entropy)

        timer_end = time.time()

        print ("-----------------")
        print ("Iteration: \t %i" % iteration)
        print ("NumTrajs: \t %i" % len(eprews))
        print ("NumTimesteps: \t %i" % np.sum(eplens))
        print ("Loss:     \t %s" % loss)
        print ("MaxRew: \t %s" % np.average([np.max(rew) for rew in all_eprews]))
        print ("MeanRew: \t %s +- %s" % (eprews.mean(), eprews.std()))
        print ("MeanSlowdown: \t %s" % np.mean(all_slowdown))
        print ("MeanLen: \t %s +- %s" % (eplens.mean(), eplens.std()))
        print ("MeanEntropy \t %s" % (np.mean(all_entropy)))
        print ("Elapsed time\t %s" % (timer_end - timer_start), "seconds")
        print ("-----------------")

        timer_start = time.time()

        max_rew_lr_curve.append(np.average([np.max(rew) for rew in all_eprews]))
        mean_rew_lr_curve.append(eprews.mean())
        slow_down_lr_curve.append(np.mean(all_slowdown))

        if iteration % pa.output_freq == 0:
            param_file = open(pa.output_filename + '_' + str(iteration) + '.pkl', 'wb')
            cPickle.dump(pg_learner.get_params(), param_file, -1)
            param_file.close()

            slow_down_cdf_xh.launch(pa, pa.output_filename + '_' + str(iteration) + '.pkl',
                                 render=False, plot=True, repre=repre, end=end)

            plot_lr_curve(pa.output_filename,
                          max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                          ref_discount_rews, ref_slow_down)


def main():

    import parameters_xh

    pa = parameters_xh.Parameters()

    pa.num_steps_in_epi = 50  # 1000
    pa.num_epis = 50  # 100
    pa.num_nw = 5
    pa.num_epis_per_batch = 20
    pa.output_freq = 50

    # pa.max_nw_size = 5
    # pa.job_len = 5
    pa.new_job_rate = 0.3

    pa.episode_max_length = 2000  # 2000

    pa.compute_dependent_parameters()

    pg_resume = None
    # pg_resume = 'data/tmp_0.pkl'

    render = False

    launch(pa, pg_resume, render, repre='image', end='all_done')


if __name__ == '__main__':
    main()
