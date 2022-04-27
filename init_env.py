import gym

def init_env(env, n_train_envs, seed):
    assert n_train_envs > 0

    if env == 'cartpole':
        env = 'CartPole-v1'
    elif env == 'acrobot':
        env = 'Acrobot-v1'
    elif env == 'mountaincar':
        env = 'MountainCar-v0'

    def make_env(env_name, env_seed):
        def init():
            temp = gym.make(env_name)
            temp.seed(env_seed)
            temp.action_space.seed(seed)
            temp.observation_space.seed(seed)
            return temp
        return init

    train_envs = gym.vector.SyncVectorEnv([make_env(env, seed + i) for i in range(n_train_envs)])
    eval_env = make_env(env, seed + n_train_envs)()
    return train_envs, eval_env
