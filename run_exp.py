import trajectory_buffer as tb
import torch as t
import numpy as np
import pandas as pd

def run_experiment(train_env, 
                    eval_env, 
                    eval_eps,
                    eval_iters, 
                    train_epochs,
                    agent, 
                    batch_size, 
                    device,
                    obs_shape,
                    gamma,
                    lam,
                    save_dirc,
                    save_iters):
    eval_results_path = save_dirc / 'eval_results.csv'
    sum_eval_results_path = save_dirc / 'sum_eval_results.csv'
    train_results_path = save_dirc / 'train_results.csv'
    train_losses_path = save_dirc / 'train_loss_results.csv'

    losses = []
    cum_rewards = []
    eps_lengths = []

    buffer = tb.TrajectoryBuffer(device, obs_shape, batch_size, gamma, lam) 
    obs = t.from_numpy(train_env.reset()).to(device)
    eps_reward = 0  # cumulative reward this episode
    eps_len = 0  # timesteps in episode so far
    for epoch in range(train_epochs):
        for j in range(batch_size):
            action, log_action_prob, entropy, value = agent.get_action_and_value(obs)
            next_obs, reward, done, _ = train_env.step(action.item())
            buffer.add(obs, action, log_action_prob, entropy, done, reward, value)

            eps_len += 1
            eps_reward += reward
            next_obs = t.from_numpy(next_obs).to(device)

            # perform batch update after `batch_size` iterations
            batch_full = j == (batch_size - 1)
            if batch_full or done:
                # bootstrap target value if next state is non-terminal 
                buffer.bootstrap(agent.get_value(next_obs) if batch_full else 0)

            # update only after a full batch
            if batch_full:
                losses.append(agent.update(buffer))
                buffer.reset()

            if done:
                cum_rewards.append(eps_reward)
                eps_lengths.append(eps_len)
                obs = t.from_numpy(train_env.reset()).to(device)
                eps_reward = 0
                eps_len = 0 

        if epoch % eval_iters == 0:
            eval_agent(eval_env, agent, eval_eps, epoch, eval_results_path, sum_eval_results_path, device)

        if (epoch % save_iters) == 0 or epoch == (train_epochs - 1):
            agent.save(save_dirc)
            pd.DataFrame({'cum_rewards': cum_rewards, 'eps_lengths': eps_lengths}).to_csv(train_results_path)
            pd.DataFrame(losses, columns=['actor', 'critic', 'entropy']).to_csv(train_losses_path)


@t.no_grad()
def eval_agent(env, agent, eval_eps, epoch, eval_results_path, sum_eval_results_path, device):
    rewards = []
    lengths = []
    for _ in range(eval_eps):
        eps_len = 0
        eps_reward = 0
        obs = env.reset()

        while True:
            action = agent.get_action(t.tensor(obs).to(device))
            next_obs, reward, done, _ = env.step(action.item())
            eps_reward += reward
            eps_len += 1
            obs = next_obs

            if done:
                rewards.append(eps_reward)
                lengths.append(eps_len)
                break
    
    epoch_eval_results = pd.DataFrame({
        'cum_reward': rewards,
        'length': lengths,
        'epoch': [epoch] * eval_eps,
        'episode': list(range(eval_eps))
    })

    rewards = np.array(rewards)
    lengths = np.array(lengths)
    sum_epoch_eval_results = pd.DataFrame([{
        'avg_cum_rwd': rewards.mean(),
        'std_cum_rwd': rewards.std(),
        'avg_len': lengths.mean(),
        'std_len': lengths.std(),
        'epoch': epoch
    }])

    if eval_results_path.exists():
        eval_results = pd.read_csv(eval_results_path, index_col=None)
        eval_results = pd.concat([eval_results, epoch_eval_results])
        eval_results.to_csv(eval_results_path, index=False)

        sum_eval_results = pd.read_csv(sum_eval_results_path, index_col=None)
        sum_eval_results = pd.concat([sum_eval_results, sum_epoch_eval_results])
        sum_eval_results.to_csv(sum_eval_results_path, index=False)
    else:
        epoch_eval_results.to_csv(eval_results_path, index=False)
        sum_epoch_eval_results.to_csv(sum_eval_results_path, index=False)