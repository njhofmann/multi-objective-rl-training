import torch as t
import scipy.signal
import numpy as np

class TrajectoryBuffer:

    def __init__(self, device, obs_shape, batch_size, gamma, lam):
        self._device = device
        self._batch_size = batch_size

        self._gamma = gamma
        self._gae_discount = self._gamma * lam
        self._obs_shape = obs_shape
        self.reset()

    @property
    def advantages(self):
        adv_std, adv_mean = t.std_mean(self._advantages)
        return (self._advantages - adv_mean) / (adv_std + 1e-8)

    def add(self, obs, action, log_action_prob, entropy, done, reward, value):
        self.observations[self._cur_idx] = obs
        self.actions[self._cur_idx] = action
        self.log_action_probs[self._cur_idx] = log_action_prob
        self.entropies[self._cur_idx] = entropy
        self.dones[self._cur_idx] = t.tensor(done).to(self._device)  # from env
        self._rewards[self._cur_idx] = t.tensor(reward).to(self._device)  # from env
        self.values[self._cur_idx] = value
        self._cur_idx += 1

    def _discounted_cum_sum(self, x, discount):
        return t.from_numpy(scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1].copy()).to(self._device)

    def bootstrap(self, bootstrap_val):
        # current trajectory info 
        start = self._cur_traj_start_idx
        end = self._cur_idx
        size = end - start + 1

        # rewards and values of the current trajectory
        rewards = np.zeros(size)#.to(self._device)
        rewards[:-1] = self._rewards[start:end].detach().cpu().numpy()
        rewards[-1] = bootstrap_val

        values = np.zeros(size)
        values[:-1] = self.values[start:end].detach().cpu().numpy()
        values[-1] = bootstrap_val

        # GAE lambda estimate
        deltas = rewards[:-1] + (self._gamma * values[1:]) - values[:-1]
        self._advantages[start:end] = self._discounted_cum_sum(deltas, self._gae_discount)

        # rewards to go, target values for critic
        self.returns[start:end] = self._discounted_cum_sum(rewards, self._gamma)[:-1]

        # set starting index of next trajectory
        self._cur_traj_start_idx = self._cur_idx
    
    def reset(self):
        self._cur_idx = 0
        self._cur_traj_start_idx = 0
        self.observations = t.zeros(( self._batch_size,) + self._obs_shape).to(self._device)
        self.actions = t.zeros( self._batch_size).to(self._device)
        self.log_action_probs = t.zeros( self._batch_size).to(self._device)
        self.entropies = t.zeros( self._batch_size).to(self._device)
        self.dones = t.zeros( self._batch_size).to(self._device)
        self._rewards = t.zeros( self._batch_size).to(self._device)
        self.values = t.zeros( self._batch_size).to(self._device)
        self._advantages = t.zeros( self._batch_size).to(self._device)
        self.returns = t.zeros( self._batch_size).to(self._device)