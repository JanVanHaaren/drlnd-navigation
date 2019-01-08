import argparse
import sys

import torch
from unityagents import UnityEnvironment

from agent_dqn import Agent


def run_agent(environment, agent):
    brain_name = environment.brain_names[0]
    environment_info = environment.reset(train_mode=False)[brain_name]
    state = environment_info.vector_observations[0]
    score = 0

    while True:
        action = agent.act(state)
        environment_info = environment.step(action)[brain_name]
        next_state = environment_info.vector_observations[0]
        reward = environment_info.rewards[0]
        done = environment_info.local_done[0]
        score += reward
        state = next_state

        if done:
            break

    print('Score: {}'.format(score))


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

    # Load the network weights
    agent.qnetwork_local.load_state_dict(torch.load(parameters.weights))

    # Run the agent
    run_agent(environment=environment, agent=agent)

    # Close the environment
    environment.close()


def parse_arguments(arguments):
    parser = argparse.ArgumentParser(description='Run the Navigation learner.')
    parser.add_argument('--weights', '-w', required=True, type=str, help='Path to a file to load the network weights.')

    return parser.parse_args(arguments)


if __name__ == '__main__':
    main(sys.argv[1:])
