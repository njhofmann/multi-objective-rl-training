import gym

def init_env(env, seed):
    if env == 'cartpole':
        train_env = gym.make('CartPole-v1')
        eval_env = gym.make('CartPole-v1')
    elif env == 'acrobot':
        train_env = gym.make('Acrobot-v1')
        eval_env = gym.make('Acrobot-v1')
    elif env == 'mountaincar':
        train_env = gym.make('MountainCar-v0')
        eval_env = gym.make('MountainCar-v0')

    train_env.seed(seed)
    train_env.action_space.seed(seed)
    train_env.observation_space.seed(seed)
    
    eval_env.seed(seed)
    eval_env.action_space.seed(seed)
    eval_env.observation_space.seed(seed)

    return train_env, eval_env
