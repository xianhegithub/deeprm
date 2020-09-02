import numpy as np
import math
import matplotlib.pyplot as plt


import parameters_xh


class Env:
    def __init__(self, pa, nw_len_epis, nw_size_epis, render=False, repre='image', end='no_new_job'):

        self.pa = pa
        self.render = render
        self.repre = repre  # image or compact representation
        self.end = end  # termination type, 'no_new_job' or 'all_done'

        self.nw_dist = pa.dist.bi_model_dist

        self.curr_time = 0

        self.nw_len_epis = nw_len_epis
        self.nw_size_epis = nw_size_epis

        self.epi_idx = 0  # which example episode
        self.job_idx_in_epi = 0  # job index in the current episode

        # initialize system
        self.machine = Machine(pa)
        self.job_slot = JobSlot(pa)
        self.job_backlog = JobBacklog(pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(pa)

    def get_new_job_from_epi(self, epi_idx, job_idx_in_epi):
        new_job = Job(res_vec=self.nw_size_epis[epi_idx, job_idx_in_epi, :],
                      job_len=self.nw_len_epis[epi_idx, job_idx_in_epi],
                      job_id=len(self.job_record.record),
                      enter_time=self.curr_time)
        return new_job

    def observe(self):

        backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))

        image_repr = np.zeros((self.pa.network_input_height, self.pa.network_input_width))

        ir_pt = 0  # somewhat a pointer

        for i in range(self.pa.num_res):

              image_repr[:, ir_pt: ir_pt + self.pa.res_slot] = self.machine.canvas[i, :, :]
              ir_pt += self.pa.res_slot  # move away from canvas, next block

              for j in range(self.pa.num_nw):

                    if self.job_slot.slot[j] is not None:  # fill in a block of work
                            image_repr[: self.job_slot.slot[j].len, ir_pt: ir_pt + self.job_slot.slot[j].res_vec[i]] = 1
                    ir_pt += self.pa.max_job_size

        image_repr[: int(self.job_backlog.curr_size / backlog_width), ir_pt: ir_pt + backlog_width] = 1
        if self.job_backlog.curr_size % backlog_width > 0:
                    image_repr[int(self.job_backlog.curr_size / backlog_width),
                    ir_pt: ir_pt + self.job_backlog.curr_size % backlog_width] = 1
        ir_pt += backlog_width

        image_repr[:, ir_pt: ir_pt + 1] = self.extra_info.time_since_last_new_job / float(
                    self.extra_info.max_tracking_time_since_last_job)
        ir_pt += 1

        assert ir_pt == image_repr.shape[1]  # debugging point

        return image_repr


    def plot_state(self):
        plt.figure("screen", figsize=(20, 5))

        skip_row = 0

        for i in range(self.pa.num_res):

            plt.subplot(self.pa.num_res,
                        1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                        i * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

            plt.imshow(self.machine.canvas[i, :, :], interpolation='nearest', vmax=1)

            for j in range(self.pa.num_nw):

                job_slot = np.zeros((self.pa.time_horizon, self.pa.max_job_size))
                if self.job_slot.slot[j] is not None:  # fill in a block of work
                    job_slot[: self.job_slot.slot[j].len, :self.job_slot.slot[j].res_vec[i]] = 1

                plt.subplot(self.pa.num_res,
                            1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                            1 + i * (
                                        self.pa.num_nw + 1) + j + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

                plt.imshow(job_slot, interpolation='nearest', vmax=1)

                if j == self.pa.num_nw - 1:
                    skip_row += 1

        skip_row -= 1
        backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
        backlog = np.zeros((self.pa.time_horizon, backlog_width))

        backlog[: self.job_backlog.curr_size / backlog_width, : backlog_width] = 1
        backlog[self.job_backlog.curr_size / backlog_width, : self.job_backlog.curr_size % backlog_width] = 1

        plt.subplot(self.pa.num_res,
                    1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                    self.pa.num_nw + 1 + 1)

        plt.imshow(backlog, interpolation='nearest', vmax=1)

        plt.subplot(self.pa.num_res,
                    1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                    self.pa.num_res * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

        extra_info = np.ones((self.pa.time_horizon, 1)) * \
                     self.extra_info.time_since_last_new_job / \
                     float(self.extra_info.max_tracking_time_since_last_job)

        plt.imshow(extra_info, interpolation='nearest', vmax=1)

        plt.show()  # manual
        # plt.pause(0.01)  # automatic

    def get_reward(self):

        reward = 0
        for j in self.machine.running_job:
            # for every time step, the agent receives for each running job a reward -1/job_length
            # so that when a job finishes, tha agent receives a reward smaller than -1
            reward += self.pa.delay_penalty / float(j.len)

        for j in self.job_slot.slot:
            if j is not None:
                reward += self.pa.hold_penalty / float(j.len)

        for j in self.job_backlog.backlog:
            if j is not None:
                reward += self.pa.dismiss_penalty / float(j.len)

        return reward

    def step(self, a, repeat=False):

        status = None

        done = False
        reward = 0
        info = None

        if a == self.pa.num_nw:  # explicit void action
            status = 'MoveOn'
        elif self.job_slot.slot[a] is None:  # implicit void action
            status = 'MoveOn'
        else:
            allocated = self.machine.allocate_job(self.job_slot.slot[a], self.curr_time)
            if not allocated:  # implicit void action
                status = 'MoveOn'
            else:
                status = 'Allocate'

        if status == 'MoveOn':
            self.curr_time += 1
            self.machine.time_proceed(self.curr_time)
            self.extra_info.time_proceed()

            # add new jobs
            self.job_idx_in_epi += 1

            if self.end == "no_new_job":  # end of new job episode
                if self.job_idx_in_epi >= self.pa.num_steps_in_epi:
                    done = True
            elif self.end == "all_done":  # everything has to be finished
                if self.job_idx_in_epi >= self.pa.num_steps_in_epi and len(self.machine.running_job) == 0 and all(
                        s is None for s in self.job_slot.slot) and all(s is None for s in self.job_backlog.backlog):
                    done = True
                elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
                    done = True

            if not done:

                if self.job_idx_in_epi < self.pa.num_steps_in_epi:  # otherwise, end of new job episode, i.e. no new jobs
                    new_job = self.get_new_job_from_epi(self.epi_idx, self.job_idx_in_epi)

                    if new_job.len > 0:  # a new job comes

                        to_backlog = True
                        for i in range(self.pa.num_nw):
                            if self.job_slot.slot[i] is None:  # put in new visible job slots
                                self.job_slot.slot[i] = new_job
                                self.job_record.record[new_job.id] = new_job
                                to_backlog = False
                                break

                        if to_backlog:
                            if self.job_backlog.curr_size < self.pa.backlog_size: #(60)
                                self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
                                self.job_backlog.curr_size += 1
                                self.job_record.record[new_job.id] = new_job
                            else:  # abort, backlog full
                                print("Backlog is full.")
                                # exit(1)

                        self.extra_info.new_job_comes()

            reward = self.get_reward()

        elif status == 'Allocate':
            self.job_record.record[self.job_slot.slot[a].id] = self.job_slot.slot[a]  # what is 'id'
            self.job_slot.slot[a] = None  # set to None because the job is sent to the machine, running.

            # dequeue backlog
            if self.job_backlog.curr_size > 0:
                self.job_slot.slot[a] = self.job_backlog.backlog[0]  # if backlog empty, it will be 0
                self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
                self.job_backlog.backlog[-1] = None
                self.job_backlog.curr_size -= 1

        ob = self.observe()

        info = self.job_record

        if done:
            self.job_idx_in_epi = 0

            if not repeat:
                self.epi_idx = (self.epi_idx + 1) % self.pa.num_epis

            self.reset()

        if self.render:
            self.plot_state()

        return ob, reward, done, info

    def reset(self):
        self.job_idx_in_epi = 0
        self.curr_time = 0

        # initialize system
        self.machine = Machine(self.pa)
        self.job_slot = JobSlot(self.pa)
        self.job_backlog = JobBacklog(self.pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(self.pa)


class Job:
    def __init__(self, res_vec, job_len, job_id, enter_time):
        self.id = job_id
        self.res_vec = res_vec
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1


class JobSlot:
    def __init__(self, pa):
        self.slot = [None] * pa.num_nw  # [None, ..., None]


class JobBacklog:
    def __init__(self, pa):
        self.backlog = [None] * pa.backlog_size  # [None]*5 is [None, None, None, None, None]
        self.curr_size = 0


class JobRecord:
    def __init__(self):
        self.record = {}


class Machine:
    def __init__(self, pa):
        self.num_res = pa.num_res
        self.time_horizon = pa.time_horizon
        self.res_slot = pa.res_slot

        self.avbl_slot = np.ones((self.time_horizon, self.num_res)) * self.res_slot

        self.running_job = []

        # colormap for graphical representation
        self.colormap = np.arange(1 / float(pa.job_num_cap), 1, 1 / float(
            pa.job_num_cap))  # job_num_cap = 40 maximum number of distinct colors in current work graph
        np.random.shuffle(self.colormap)

        # graphical representation
        self.canvas = np.zeros((pa.num_res, pa.time_horizon, pa.res_slot))

    def allocate_job(self, job, curr_time):

        allocated = False

        for t in range(0, self.time_horizon - job.len):

            new_avbl_res = self.avbl_slot[t: t + job.len, :] - job.res_vec

            if np.all(new_avbl_res[:] >= 0):

                allocated = True

                self.avbl_slot[t: t + job.len, :] = new_avbl_res
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len

                self.running_job.append(job)

                # update graphical representation

                used_color = np.unique(self.canvas[:])
                # WARNING: there should be enough colors in the color map
                # find the next color in the colormap for the new job
                for color in self.colormap:
                    if color not in used_color:
                        new_color = color
                        break

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time
                canvas_start_time = job.start_time - curr_time  # should be just t
                canvas_end_time = job.finish_time - curr_time  # should be t + job.len

                for res in range(self.num_res):
                    for i in range(canvas_start_time, canvas_end_time):
                        avbl_slot = np.where(self.canvas[res, i, :] == 0)[
                            0]  # find the location of the available resource slot in the time step t
                        self.canvas[res, i, avbl_slot[: job.res_vec[res]]] = new_color

                break

        return allocated

    def time_proceed(self, curr_time):
        # update the machine state!
        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
        self.avbl_slot[-1, :] = self.res_slot
        # check whether jobs are finished at this point
        for job in self.running_job:

            if job.finish_time <= curr_time:
                self.running_job.remove(job)  # an attribute of list, .remove or .append elements.

        # update graphical representation

        self.canvas[:, :-1, :] = self.canvas[:, 1:,
                                 :]  # index starts from 0, this line removes the first element of canvas in the second dimension.
        self.canvas[:, -1, :] = 0


class ExtraInfo:
    def __init__(self, pa):
        self.time_since_last_new_job = 0
        self.max_tracking_time_since_last_job = pa.max_track_since_new

    def new_job_comes(self):
        self.time_since_last_new_job = 0

    def time_proceed(self):
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1


# ==========================================================================
# ------------------------------- Unit Tests -------------------------------
# ==========================================================================


def test_backlog():
    pa = parameters_xh.Parameters()
    pa.num_nw = 5
    pa.num_steps_in_epi = 50
    pa.num_epis = 10
    pa.new_job_rate = 1
    pa.compute_dependent_parameters()

    import job_distribution_xh

    nw_len_epis, nw_size_epis = job_distribution_xh.generate_episodes_work(pa)
    env = Env(pa, nw_len_epis=nw_len_epis, nw_size_epis=nw_size_epis, render=False, repre='image')

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    env.step(5)
    assert env.job_backlog.backlog[0] is not None
    assert env.job_backlog.backlog[1] is None
    print("New job is backlogged.")

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(3)
    assert env.job_slot.slot[3] == job

    print("- Backlog test passed -")


def test_compact_speed():
    pa = parameters_xh.Parameters()
    pa.num_steps_in_epi = 50
    pa.num_epis = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    import job_distribution_xh

    nw_len_epis, nw_size_epis = job_distribution_xh.generate_episodes_work(pa)
    env = Env(pa, nw_len_epis=nw_len_epis, nw_size_epis=nw_size_epis, render=False, repre='compact')

    import other_agents
    import time

    start_time = time.time()
    for i in range(100000):
        a = other_agents.get_sjf_action(env.machine, env.job_slot)
        env.step(a)
    end_time = time.time()
    print("- Elapsed time: ", end_time - start_time, "sec -")


def test_image_speed():
    pa = parameters_xh.Parameters()
    pa.num_steps_in_epi = 50
    pa.num_epis = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    import job_distribution_xh

    nw_len_epis, nw_size_epis = job_distribution_xh.generate_episodes_work(pa)
    env = Env(pa, nw_len_epis=nw_len_epis, nw_size_epis=nw_size_epis, render=False, repre='image')

    import other_agents
    import time

    start_time = time.time()
    for i in range(100000):
        a = other_agents.get_sjf_action(env.machine, env.job_slot)
        env.step(a)
    end_time = time.time()
    print("- Elapsed time: ", end_time - start_time, "sec -")


if __name__ == '__main__':
    test_backlog()
    test_compact_speed()
    test_image_speed()
