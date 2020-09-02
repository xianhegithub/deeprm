import numpy as np


class Dist:

    def __init__(self, num_res, max_nw_size, job_len):
        self.num_res = num_res
        self.max_nw_size = max_nw_size    #max_job_size in parameters.py 10
        self.job_len = job_len      #max_job_len in parameters.py 15

        self.job_small_chance = 0.8

        self.job_len_big_lower = job_len * 2 / 3        #10
        self.job_len_big_upper = job_len        #15

        self.job_len_small_lower = 1
        self.job_len_small_upper = job_len / 5      #3  == 3t

        self.dominant_res_lower = max_nw_size / 2       #5
        self.dominant_res_upper = max_nw_size       #10  == 0.5r, r = 20

        self.other_res_lower = 1
        self.other_res_upper = max_nw_size / 5      #2  ==0.1r

    def normal_dist(self):

        # new work duration
        nw_len = np.random.randint(1, self.job_len + 1)  # same length in every dimension

        nw_size = np.zeros(self.num_res)

        for i in range(self.num_res):
            nw_size[i] = np.random.randint(1, self.max_nw_size + 1)

        return nw_len, nw_size

    def bi_model_dist(self):
        # -- generate individual job --
        # -- job length --
        if np.random.rand() < self.job_small_chance:  # small job
            nw_len = np.random.randint(self.job_len_small_lower,
                                       self.job_len_small_upper + 1)
        else:  # big job
            nw_len = np.random.randint(self.job_len_big_lower,
                                       self.job_len_big_upper + 1)

        nw_size = np.zeros(self.num_res)

        # -- job resource request --
        dominant_res = np.random.randint(0, self.num_res)
        for i in range(self.num_res):
            if i == dominant_res:
                nw_size[i] = np.random.randint(self.dominant_res_lower,
                                               self.dominant_res_upper + 1)
            else:
                nw_size[i] = np.random.randint(self.other_res_lower,
                                               self.other_res_upper + 1)

        return nw_len, nw_size


def generate_episodes_work(pa):

    if pa.unseen:
        np.random.seed(314159)
    else:
        np.random.seed(42)
    # pa.num_steps_in_epi: length of the busy cycle that repeats itself. pa.num_epis: number of episodes
    total_num_steps = pa.num_steps_in_epi * pa.num_epis   # like a clock, each step the machine checks whether there is a new job coming

    nw_dist = pa.dist.bi_model_dist

    nw_len_epis = np.zeros(total_num_steps, dtype=int)
    nw_size_epis = np.zeros((total_num_steps, pa.num_res), dtype=int)

    for i in range(total_num_steps):

        if np.random.rand() < pa.new_job_rate:  # a new job comes

            nw_len_epis[i], nw_size_epis[i, :] = nw_dist()

    workload = np.zeros(pa.num_res)
    for i in range(pa.num_res):
        # res_slot maximum number of available slots
        # roughly the average work load per slot per step
        workload[i] = np.sum(nw_size_epis[:, i] * nw_len_epis) / float(pa.res_slot) / float(
            len(nw_len_epis))
        print("Load on # " + str(i) + " resource dimension is " + str(workload[i]))

    nw_len_epis = np.reshape(nw_len_epis,
                            [pa.num_epis, pa.num_steps_in_epi])
    nw_size_epis = np.reshape(nw_size_epis,
                             [pa.num_epis, pa.num_steps_in_epi, pa.num_res])

    return nw_len_epis, nw_size_epis
