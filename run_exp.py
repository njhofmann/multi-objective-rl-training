import trajectory_buffer as tb
import torch as t
import numpy as np
import pandas as pd

def run_experiment(train_envs, 
                    eval_env, 
                    eval_eps,
                    eval_iters, 
                    train_epochs,
                    agent, 
                    batch_size, 
                    device,
                    obs_shape,
                    gamma,
                    gae_lambda,
                    save_dirc,
                    save_iters,
                    n_envs):
    eval_results_path = save_dirc / 'eval_results.csv'
    sum_eval_results_path = save_dirc / 'sum_eval_results.csv'
    train_results_path = save_dirc / 'train_results.csv'
    train_losses_path = save_dirc / 'train_loss_results.csv'

    losses = []
    train_eps_info = []
    cum_rewards = np.zeros(n_envs)
    eps_lengths = np.zeros(n_envs)

    buffer = tb.TrajectoryBuffer(device, obs_shape, n_envs, batch_size, gamma, gae_lambda) 
    obs = t.from_numpy(train_envs.reset()).to(device)
    timestep = 0
    for epoch in range(train_epochs):
        for j in range(batch_size):
            actions, log_action_probs, entropies, values = agent.get_actions_and_values(obs)
            next_obs, rewards, dones, _ = train_envs.step(actions.cpu().numpy())
            buffer.add(obs, actions, log_action_probs, entropies, dones, rewards, values)

            eps_lengths += 1
            cum_rewards += rewards
            next_obs = t.from_numpy(next_obs).to(device)
            timestep += 1

            # bootstrap current trajectory for environments when:
            # - `batch_size` steps have occured (ie buffer is full)
            # - any environment(s) has terminated
            batch_full = (j == (batch_size - 1))
            if batch_full or dones.any():
                # target value is bootstraped for non-terminal environments
                bootstrap_vals = agent.get_values(next_obs)

                 # terminated environments have bootstrap value of 0 
                bootstrap_vals[dones] = 0.0 

                # perform updates only on all environments only if buffer is full
                to_bootstrap = np.full(n_envs, True) if batch_full else dones      
                buffer.bootstrap(bootstrap_vals, to_bootstrap)

            # update only after a full batchaa
            if batch_full:
                losses.append(agent.update(buffer))
                buffer.reset()

            for i, done in enumerate(dones):
                if done:
                    # training timestep, environment ID, cumulative reward, episode length
                    train_eps_info.append((timestep, i, cum_rewards[i], eps_lengths[i]))

                    # don't need to restart environment, handled behind the scenes
                    cum_rewards[i] = 0.0
                    eps_lengths[i] = 0.0

        if epoch % eval_iters == 0:
            eval_agent(eval_env, agent, eval_eps, epoch, eval_results_path, sum_eval_results_path, device)

        if (epoch % save_iters) == 0 or epoch == (train_epochs - 1):
            agent.save(save_dirc)
            pd.DataFrame(train_eps_info, columns=['timestep', 'env', 'cum_reward', 'eps_len']).to_csv(train_results_path)
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
            obs = t.tensor([obs]).to(device)
            action = agent.get_actions(obs)
            next_obs, reward, done, _ = env.step(action.item())
            eps_reward += reward
            eps_len += 1
            obs = next_obs

            if done:
                rewards.append(eps_reward)
                lengths.append(eps_len)
                break
    
    epoch_eval_results = pd.DataFrame({
        'cum_rwd': rewards,
        'len': lengths,
        'epoch': [epoch] * eval_eps,
        'eps': list(range(eval_eps))
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