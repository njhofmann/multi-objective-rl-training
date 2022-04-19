import argparse as ap
import random as r
import numpy as np
import torch as t
import init_env as ie
import run_exp
import gym
import actor_critic as ac
import pathlib as pl
import datetime as dt
import yaml as y

def init_save_dirc(parent_dirc):
    parent_dirc = pl.Path(parent_dirc)
    parent_dirc.mkdir(exist_ok=True, parents=True)

    exp_dirc_name = str(dt.datetime.now()).replace(' ', '_').replace('.', '_')
    save_dirc = parent_dirc / exp_dirc_name
    save_dirc.mkdir()
    return save_dirc


def set_up(seed):
    r.seed(seed)
    np.random.seed(seed)
    t.manual_seed(0)
    t.use_deterministic_algorithms(True)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False

    if t.cuda.is_available():
        return t.device('cuda')
    raise RuntimeError('CUDA not available')


def get_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--save-dirc', type=str, default='exp_results', help='directory to store results of all experiments')
    parser.add_argument('--save-iters', type=int, default=10000, help='save models every X policy updates')
    parser.add_argument('--seed', type=int, default=7, help='seed for all random procsses')
    parser.add_argument('--env', type=str, default=None, help='environment for agent to learn')
    parser.add_argument('--agent-method', type=str, default=None, help='type of PPO agent:\n' \
                                                                        '- separate: actor & critic have their own network\n' \
                                                                        '- shared: actor & critic share a single network')
    # TODO expand this
    parser.add_argument('--train-method', type=str, default=None, help='training regime to use:\n' \
                                                                        '- linear-sum: losses are treated as a linear sum, \n' \
                                                                        '- mmdm: modified differential method of multipliers')
    parser.add_argument('--eval-eps', type=int, default=20, help='number of evaluation episodes to run during each evaluation cycle')
    parser.add_argument('--eval-iters', type=int, default=1000, help='run an evaluation cycle every X policy updates')
    parser.add_argument('--n-updates', type=int, default=33000, help='number of policy updates to perform')
    parser.add_argument('--batch-size', type=int, default=32, help='number of interactions making up data for a policy update (i.e. the minibatch)')
    parser.add_argument('--gamma', type=float, default=.99, help='discount factor for rewards')
    parser.add_argument('--lam', type=float, default=.97, help='lambda argument for GAE-lambda estimation')
    parser.add_argument('--actor-arch', type=int, nargs='+', default=[64, 64], help='layers width for actor network, used only if `train_method` set to `separate`')
    parser.add_argument('--critic-arch', type=int, nargs='+', default=[64, 64], help='layers widths for critic network, used only if `train_method` set to `separate`')
    parser.add_argument('--arch', type=int, nargs='+', default=[64, 64], help='layers widths for shared actor-critic network, used only if `train_method` set to `shared`')
    parser.add_argument('--actor-lr', type=float, default=.0003, help='learning rate for actor network, used only if `train_method` set to `separate`')
    parser.add_argument('--critic-lr', type=float, default=.001, help='learning rate for shared actor-critic network, used only if `train_method` set to `shared`')
    parser.add_argument('--lr', type=float, default=.0007, help='learning rate for critic network, used only if `train_method` set to `shared`')
    parser.add_argument('--critic-weight', type=float, default=.25, help='weight for critic network loss, used only if `training_method` set to `shared`')
    parser.add_argument('--entropy-weight', type=float, default=.001, help='weight for entropy loss for actor network')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = set_up(args.seed)
    train_env, eval_env = ie.init_env(args.env, args.seed)
    save_dirc = init_save_dirc(args.save_dirc)

    if not isinstance(train_env.action_space, gym.spaces.Discrete):
        raise ValueError('only support environments with discrete action spaces')
    n_actions = train_env.action_space.n

    with open(save_dirc / 'params.yaml', 'w') as f:
        y.dump(vars(args), f)
    
    agent = ac.ActorCritic(init_method=args.agent_method, 
                            train_method=args.train_method,
                            n_actions=n_actions,
                            shared_arch=args.arch,
                            actor_arch=args.actor_arch,
                            critic_arch=args.critic_arch,
                            obs_space=train_env.observation_space.shape,
                            shared_lr=args.lr,
                            actor_lr=args.actor_lr,
                            critic_lr=args.critic_lr,
                            critic_loss_weight=args.critic_weight,
                            entropy_loss_weight=args.entropy_weight
                           ).to(device)

    run_exp.run_experiment(train_env=train_env,
                            eval_env=eval_env,
                            eval_eps=args.eval_eps,
                            eval_iters=args.eval_iters,
                            train_epochs=args.n_updates,
                            agent=agent,
                            batch_size=args.batch_size,
                            device=device,
                            obs_shape=train_env.observation_space.shape,
                            gamma=args.gamma,
                            lam=args.lam,
                            save_dirc=save_dirc,
                            save_iters=args.save_iters)
