import argparse
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from agent_dqn import Agent


def run_dqn(environment, agent, weights, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Run Deep Q-Learning for the given agent in the given environment.
    
    Params
    ======
        environment (UnityEnvironment): Environment to run the agent in
        agent (Agent): Agent to run in the environment
        weights (str): Path to file to store the network weights
        n_episodes (int): Maximum number of training episodes
        max_t (int): Maximum number of timesteps per episode
        eps_start (float): Starting value of epsilon for epsilon-greedy action selection
        eps_end (float): Minimum value of epsilon
        eps_decay (float): Multiplicative factor (per episode) for decreasing epsilon
    """

    brain_name = environment.brain_names[0]

    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    for i_episode in range(1, n_episodes + 1):
        environment_info = environment.reset(train_mode=True)[brain_name]
        state = environment_info.vector_observations[0]
        score = 0

        for t in range(max_t):
            action = agent.act(state, eps)
            environment_info = environment.step(action)[brain_name]
            next_state = environment_info.vector_observations[0]
            reward = environment_info.rewards[0]
            done = environment_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        print('\rEpisode {}\tAverage score: {:.2f}'.format(i_episode, np.mean(scores_window)), end='')
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        if np.mean(scores_window) >= 18.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage score: {:.2f}'.format(i_episode - 100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), weights)
            break

    return scores


def plot_scores(scores, plot_name):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.savefig(plot_name)
    plt.show()


def setup_environment(file_name):
    environment = UnityEnvironment(file_name=file_name)
    brain_name = environment.brain_names[0]
    brain = environment.brains[brain_name]
    environment_info = environment.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = len(environment_info.vector_observations[0])

    return environment, action_size, state_size


def main(arguments):
    parameters = parse_arguments(arguments)

    # Specify the path to the environment
    file_name = 'Banana_Linux/Banana.x86_64'

    # Set up the environment
    environment, action_size, state_size = setup_environment(file_name)

    # Set up the agent
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    # Run the Deep Q-Learning algorithm
    scores = run_dqn(environment=environment, agent=agent, weights=parameters.weights, n_episodes=parameters.episodes)

    # Plot the scores
    plot_scores(scores, parameters.plot)

    # Close the environment
    environment.close()


def parse_arguments(arguments):
    parser = argparse.ArgumentParser(description='Run the Navigation learner.')
    parser.add_argument('--episodes', '-e', required=True, type=int, help='Number of episodes to run.')
    parser.add_argument('--weights', '-w', required=True, type=str, help='Path to a file to store the network weights.')
    parser.add_argument('--plot', '-p', required=True, type=str, help='Path to a file to store a plot of the scores.')

    return parser.parse_args(arguments)


if __name__ == '__main__':
    main(sys.argv[1:])
