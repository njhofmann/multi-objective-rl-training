import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch as t
import itertools as it
import torchinfo 

class ActorCritic(nn.Module):

    def __init__(self, 
                 device,
                 n_envs,
                 init_method, 
                 train_method,
                 n_actions, 
                 obs_space, 
                 shared_arch,
                 actor_arch, 
                 critic_arch,
                 shared_lr,
                 actor_lr,
                 critic_lr,
                 critic_loss_weight,
                 entropy_loss_weight,
                 max_grad_norm,
                 mdmm_epsilon,
                 mdmm_damping):
        super().__init__()
        self._device = device
        self._init_method = init_method
        self._train_method = train_method
        self._critic_loss_weight = critic_loss_weight
        self._entropy_loss_weight = entropy_loss_weight
        self._max_grad_norm = max_grad_norm
        self._mdmm_lambda = t.tensor(0.0, requires_grad=True)
        self._mdmm_epsilon = mdmm_epsilon
        self._mdmm_damping = mdmm_damping

        self._init_networks(shared_arch, actor_arch, critic_arch, n_actions, obs_space, n_envs)        
        self._init_optimizers(shared_lr, actor_lr, critic_lr)

    def _init_optimizers(self, shared_lr, actor_lr, critic_lr):
        if self._init_method == 'shared':
            self._optimizer = t.optim.Adam(self.parameters(), lr=shared_lr)
        elif self._init_method == 'separate':
            self._actor_optimizer = t.optim.Adam(self._actor.parameters(), lr=actor_lr)
            self._critic_optimizer = t.optim.Adam(self._critic.parameters(), lr=critic_lr)

    def _init_network(self, arch, mark_last=True):
        n_layers = len(arch)
        return nn.Sequential(*list(it.chain(*[(nn.Linear(arch[i-1], arch[i]), nn.Identity() if i == n_layers - 1 and mark_last else nn.ReLU()) 
                                for i in range(1, n_layers)])))

    def _init_networks(self, shared_arch, actor_arch, critic_arch, n_actions, obs_space, n_envs):
        if len(obs_space) == 1:
            obs_space = obs_space[0]
        action_shape = (n_envs, n_actions)
        obs_shape = (n_envs, obs_space)

        if self._init_method == 'shared':
            shared = self._init_network([obs_space, *shared_arch], mark_last=False)
            self._actor = t.nn.Sequential(shared, nn.Linear(in_features=shared_arch[-1], out_features=n_actions))
            self._critic = t.nn.Sequential(shared, nn.Linear(in_features=shared_arch[-1], out_features=1))
        elif self._init_method == 'separate':
            self._actor = self._init_network([obs_space, *actor_arch, n_actions])
            self._critic = self._init_network([obs_space, *critic_arch, 1])
        else:
            raise ValueError(f'{self._init_method} is not a supported initalization method')

        torchinfo.summary(self._actor, input_size=obs_shape)
        torchinfo.summary(self._critic, input_size=obs_shape)

    def get_actions_and_values(self, observation):
        action_probs = self._get_action_probs(observation)
        action = action_probs.sample()
        action_prob = action_probs.log_prob(action)
        entropy = action_probs.entropy()
        value = self.get_values(observation)
        return action, action_prob, entropy, value

    def get_actions(self, observation):  
        return self._get_action_probs(observation).sample()

    def _get_action_probs(self, obs):
        return Categorical(logits=self._actor(obs))

    def get_values(self, obs):
        return t.squeeze(self._critic(obs), -1)

    def update(self, buffer):
        actor_loss = self._compute_actor_loss(buffer.advantages, buffer.log_action_probs)
        critic_loss = self._compute_critic_loss(buffer.returns, buffer.values)
        entropy_loss = self._compute_entropy_loss(buffer.entropies)

        if self._init_method == 'shared':
            # optimize a shared loss
            self._optimizer.zero_grad()

            if self._train_method == 'linear-sum':
                loss = actor_loss + (self._critic_loss_weight * critic_loss) + (self._entropy_loss_weight * entropy_loss)
                loss.backward()
            elif self._train_method == 'mdmm':
                damp = self._mdmm_damping * (self._mdmm_epsilon - critic_loss).detach() # + (self._entropy_loss_weight * entropy_loss)
                loss = actor_loss - ((self._mdmm_lambda - damp) * (self._mdmm_epsilon - critic_loss))
                loss.backward()
                
                self._mdmm_lambda = t.tensor(self._mdmm_lambda + (1.0 * self._mdmm_lambda.grad), requires_grad=True) # TODO why is grad None after second call
                self._mdmm_lambda = max(self._mdmm_lambda, t.tensor(0.0, requires_grad=True))

            
            nn.utils.clip_grad_norm_(self.parameters(), self._max_grad_norm)
            self._optimizer.step()

        elif self._init_method == 'separate':
            # optimize actor
            self._actor_optimizer.zero_grad()
            loss = actor_loss + (self._entropy_loss_weight * entropy_loss)
            loss.backward()  
            nn.utils.clip_grad_norm_(self._actor.parameters(), self._max_grad_norm)
            self._actor_optimizer.step()
                
            # optimize critic
            self._critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self._critic.parameters(), self._max_grad_norm)
            self._critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), entropy_loss.item()

    def _compute_actor_loss(self, advantages, log_probs):
        return -(advantages * log_probs).mean()

    def _compute_critic_loss(self, returns, values):
        return ((returns - values) ** 2).mean()

    def _compute_entropy_loss(self, entropies):
        return -1 * entropies.mean()

    def save(self, dirc_path):
        t.save(self._actor.state_dict(), dirc_path / 'actor.pth')
        t.save(self._critic.state_dict(), dirc_path / 'critic.pth')
