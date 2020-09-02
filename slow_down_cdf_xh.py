import numpy as np
import matplotlib.pyplot as plt
import scipy

import job_distribution_xh
import environment_xh
import parameters_xh
import pg_network_xh
import other_policies_xh


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    assert x.ndim >= 1
    # More efficient version:
    out = scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out  # output np.array y[i]


def categorical_sample(prob_n):
    """
    Sample from categorical distribution,
    specified by a vector of class probabilities
    This function can potentially be replace by np.random.choice()
    """
    prob_n = np.asarray(prob_n)
    return np.random.choice(len(prob_n),p = prob_n)


def get_traj(test_type, pa, env, episode_max_length, pg_resume=None, render=False):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """


    env.reset()
    rews = []
    state =np.zeros(shape=[1,1,20,224])

    ob = env.observe()

    for _ in range(episode_max_length):

        if test_type == 'PG':
            state[0, 0, :, :] = ob
            act_prob =pg_resume.predict(state)
            a = np.random.choice(len(np.asarray(act_prob)), p=np.asarray(act_prob))

        elif test_type == 'Tetris':
            a = other_policies_xh.get_packer_action(env.machine, env.job_slot)

        elif test_type == 'SJF':
            a = other_policies_xh.get_sjf_action(env.machine, env.job_slot)

        elif test_type == 'Random':
            a = other_policies_xh.get_random_action(env.job_slot)

        ob, rew, done, info = env.step(a, repeat=True)

        rews.append(rew)

        if done: break
        if render: env.render()
        # env.render()

    return np.array(rews), info


def launch(pa, pg_resume=None, render=False, plot=False, repre='image', end='no_new_job'):

    # ---- Parameters ----

    test_types = ['Tetris', 'SJF', 'Random']

    if pg_resume is not None:
        test_types = ['PG'] + test_types

    nw_len_epis, nw_size_epis = job_distribution_xh.generate_episodes_work(pa)
    env = environment_xh.Env(pa, nw_len_epis=nw_len_epis, nw_size_epis=nw_size_epis, render=False, repre=repre, end=end)

    # these are several disctionaries, indexed by the string test_types: ['Tetris','SJF','Random']
    all_discount_rews = {}
    jobs_slow_down = {}
    work_complete = {}
    work_remain = {}
    job_len_remain = {}
    num_job_remain = {}
    job_remain_delay = {}

    for test_type in test_types:
        all_discount_rews[test_type] = []
        jobs_slow_down[test_type] = []
        work_complete[test_type] = []
        work_remain[test_type] = []
        job_len_remain[test_type] = []
        num_job_remain[test_type] = []
        job_remain_delay[test_type] = []

    for epi_idx in range(pa.num_epis):  # number of episodes
        print('\n\n')
        print("=============== " + str(epi_idx) + " ===============")

        for test_type in test_types:

            rews, info = get_traj(test_type, pa, env, pa.episode_max_length, pg_resume)

            print ("---------- " + test_type + " -----------")

            print ("total discount reward : \t %s" % (discount(rews, pa.discount)[0]))

            all_discount_rews[test_type].append(
                discount(rews, pa.discount)[0]  # this is just the discounted reward
            )

            # ------------------------
            # ---- per job stat ----
            # ------------------------

            enter_time = np.array([info.record[i].enter_time for i in range(len(info.record))])
            finish_time = np.array([info.record[i].finish_time for i in range(len(info.record))])
            job_len = np.array([info.record[i].len for i in range(len(info.record))])
            job_total_size = np.array([np.sum(info.record[i].res_vec) for i in range(len(info.record))])

            finished_idx = (finish_time >= 0)
            unfinished_idx = (finish_time < 0)

            jobs_slow_down[test_type].append(
                (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]
            )
            work_complete[test_type].append(
                np.sum(job_len[finished_idx] * job_total_size[finished_idx])
            )
            work_remain[test_type].append(
                np.sum(job_len[unfinished_idx] * job_total_size[unfinished_idx])
            )
            job_len_remain[test_type].append(
                np.sum(job_len[unfinished_idx])
            )
            num_job_remain[test_type].append(
                len(job_len[unfinished_idx])
            )
            job_remain_delay[test_type].append(
                np.sum(pa.episode_max_length - enter_time[unfinished_idx])
            )

        env.epi_idx = (env.epi_idx + 1) % env.pa.num_epis

    # -- matplotlib colormap no overlap --
    if plot:
        num_colors = len(test_types)
        cm = plt.get_cmap('gist_rainbow')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        col = [cm(1. * i / num_colors) for i in range(num_colors)]
        #ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

        for test_index,test_type in enumerate(test_types):
            # the following two lines calculate the cumulative distribution function
            slow_down_cdf = np.sort(np.concatenate(jobs_slow_down[test_type]))
            slow_down_yvals = np.arange(len(slow_down_cdf))/float(len(slow_down_cdf))
            ax.plot(slow_down_cdf, slow_down_yvals, linewidth=2, label=test_type, color = col[test_index])

        plt.legend(loc=4)
        plt.xlabel("job slowdown", fontsize=20)
        plt.ylabel("CDF", fontsize=20)
        plt.show()
        #plt.savefig(pg_resume + "_slowdown_fig" + ".pdf")

    return all_discount_rews, jobs_slow_down


def main():
    pa = parameters_xh.Parameters()

    pa.num_steps_in_epi = 200  # 5000  # 1000
    pa.num_epis = 10  # number of episodes, this is basically number of episode.
    pa.num_nw = 10
    pa.num_epis_per_batch = 20  # number of episodes to compute baseline
    # pa.max_nw_size = 5
    # pa.job_len = 5
    pa.new_job_rate = 0.3
    pa.discount = 1

    pa.episode_max_length = 20000  # 2000

    pa.compute_dependent_parameters()

    render = False

    plot = True  # plot slowdown cdf

    pg_resume = None
    #pg_resume = 'data/pg_re_discount_1_rate_0.3_simu_len_200_num_seq_per_batch_20_ex_10_nw_10_1450.pkl'
    # pg_resume = 'data/pg_re_1000_discount_1_5990.pkl'

    pa.unseen = True

    launch(pa, pg_resume, render, plot, repre='image', end='all_done')


if __name__ == '__main__':
    main()
