import torch as t
import scipy.signal
import numpy as np

class TrajectoryBuffer:

    def __init__(self, device, obs_shape, n_envs, batch_size, gamma, lam):
        self._device = device
        self._batch_size = batch_size
        self._n_envs = n_envs
        self._gamma = gamma
        self._gae_discount = self._gamma * lam
        self._obs_shape = obs_shape
        self._init = False
        self.reset()

    @property
    def advantages(self):
        advantages = self._advantages.view(-1)
        adv_std, adv_mean = t.std_mean(advantages)
        return (advantages - adv_mean) / (adv_std + 1e-8)

    @property
    def returns(self):
        return self._returns.view(-1)

    @property
    def values(self):
        return self._values.view(-1)

    @property
    def log_action_probs(self):
        return self._log_action_probs.view(-1)

    @property
    def entropies(self):
        return self._entropies.view(-1)

    def add(self, obs, action, log_action_prob, entropy, done, reward, value):
        self._observations[self._cur_idx] = obs
        self._actions[self._cur_idx] = action
        self._log_action_probs[self._cur_idx] = log_action_prob
        self._entropies[self._cur_idx] = entropy
        self._dones[self._cur_idx] = t.tensor(done).to(self._device)  # from env
        self._rewards[self._cur_idx] = t.tensor(reward).to(self._device)  # from env
        self._values[self._cur_idx] = value
        self._cur_idx += 1

    def _discounted_cum_sum(self, x, discount):
        return t.from_numpy(scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1].copy()).to(self._device)

    def bootstrap(self, bootstrap_vals, envs_to_bootstrap):
        assert len(envs_to_bootstrap) == self._n_envs

        for i in range(self._n_envs):
            if envs_to_bootstrap[i]:
                # current trajectory info 
                start = self._cur_traj_start_idx[i]
                end = self._cur_idx  # TODO can this be one value?
                size = end - start + 1

                bootstrap_val = bootstrap_vals[i]

                # rewards and values of the current trajectory
                rewards = np.zeros(size)
                rewards[:-1] = self._rewards[start:end, i].detach().cpu().numpy()
                rewards[-1] = bootstrap_val

                values = np.zeros(size)
                values[:-1] = self._values[start:end, i].detach().cpu().numpy()
                values[-1] = bootstrap_val

                # GAE lambda estimate
                deltas = rewards[:-1] + (self._gamma * values[1:]) - values[:-1]
                self._advantages[start:end, i] = self._discounted_cum_sum(deltas, self._gae_discount)

                # rewards to go, target values for critic
                self._returns[start:end, i] = self._discounted_cum_sum(rewards, self._gamma)[:-1]

                # set starting index of next trajectory
                self._cur_traj_start_idx[i] = self._cur_idx
    
    def reset(self):
        # don't want to assert on first call, first call initalizes all variables
        if self._init: 
            # should only be called after buffer is full
            print(self._cur_idx)
            assert self._cur_idx == self._batch_size 
            assert all(self._cur_traj_start_idx == self._batch_size)
        else:
            self._init = True

        # current step for current trajectory of i-th environment
        self._cur_idx = 0

        # index where current trajectory for i-th environment starts
        self._cur_traj_start_idx = np.zeros(self._n_envs, dtype=np.int32)

        self._observations = t.zeros((self._batch_size, self._n_envs) + self._obs_shape).to(self._device)
        self._actions = t.zeros((self._batch_size, self._n_envs)).to(self._device)
        self._log_action_probs = t.zeros((self._batch_size, self._n_envs)).to(self._device)
        self._entropies = t.zeros((self._batch_size, self._n_envs)).to(self._device)
        self._dones = t.zeros((self._batch_size, self._n_envs)).to(self._device)
        self._rewards = t.zeros((self._batch_size, self._n_envs)).to(self._device)
        self._values = t.zeros((self._batch_size, self._n_envs)).to(self._device)
        self._advantages = t.zeros((self._batch_size, self._n_envs)).to(self._device)
        self._returns = t.zeros((self._batch_size, self._n_envs)).to(self._device)