import torch as t
import scipy

class TrajectoryBuffer:

    def __init__(self, device, obs_shape, batch_size, gamma, lam):
        self._device = device
        self._batch_size = batch_size

        self.observations = t.zeros((batch_size,) + obs_shape).to(device)
        self.actions = t.zeros(batch_size).to(device)
        self.entropies = t.zeros((batch_size,)).to(device)
        self.log_probs = t.zeros(batch_size).to(device)
        self.dones = t.zeros(batch_size).to(device)     
        self.rewards = t.zeros(batch_size).to(device)  
        self.values = t.zeros(batch_size).to(device) 
        self._advantages = t.zeros(batch_size).to(device)

        self._gamma = gamma
        self._lambda = lam

        self._cur_idx = 0
        self._max_size = batch_size
        self._cur_traj_start_idx = 0  # index of where current trajectory started

    @property
    def advantages(self):
        adv_std, adv_mean  = t.std_mean(self._advantages)
        return (self._advantages - adv_mean) / adv_std

    def add(self, obs, action, entropy, log_prob, done, reward, value):
        # TODO send to device
        self.observations[self._cur_idx] = obs
        self.actions[self._cur_idx] = action
        self.entropies[self._cur_idx] = entropy 
        self.log_probs[self._cur_idx] = log_prob
        self.dones[self._cur_idx] = done
        self.rewards[self._cur_idx] = reward
        self.values[self._cur_idx] = value
        self._cur_idx += 1

    def _discounted_cum_sum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def bootstrap(self, bootstrap_val):
        # current trajectory info 
        start = self._cur_traj_start_idx
        end = self._cur_idx
        size = end - start + 1

        # rewards and values of the current trajectory
        rewards = t.zeros(size).to(self._device)
        rewards[start:end] = self.rewards[start:end]
        rewards[-1] = bootstrap_val

        values = t.zeros(size).to(self._device)
        values[start:end] = self.values[start:end]
        values[-1] = bootstrap_val

        # GAE lambda estimate
        deltas = rewards[:-1] + (self._gamma * values[1:]) - values[:-1]
        self._advantages[start:end] = self._discounted_cum_sum(deltas, self._gamma * self._lambda)

        # rewards to go, target values for critic
        self.rewards[start:end] = self._discounted_cum_sum(rewards, self._gamma)[:-1]

        # set starting index of next trajectory
        self._cur_traj_start_idx = self._cur_idx
    
    def reset(self):
        self._cur_idx = 0
        self._cur_traj_start_idx = 0