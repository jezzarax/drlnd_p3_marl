import json
import logging
import os
import sys

from unityagents import UnityEnvironment

from agents import *
from models import *

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    stream=sys.stdout
)

ENVIRONMENT_BINARY = os.environ['DRLUD_P3_ENV']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path_prefix = "./hp_search_results/"


def train(agent, environment, n_episodes=5000, max_t=20000, random_action_phase_enabled=True, random_episodes_number=200, store_weights_to="checkpoint.pth"):
    scores = []  # list containing scores from each episode
    for i_episode in range(1, n_episodes + 1):
        env_info = environment.reset(train_mode=True)[agent.name]
        states = env_info.vector_observations
        episode_scores = []
        episode_rewards = []
        for t in range(max_t):
            if random_action_phase_enabled and i_episode < random_episodes_number:
                actions = np.random.randn(len(env_info.agents), agent.config.action_size) # select an action (for each agent)
            else:
                actions = agent.act(states)
            actions = np.clip(actions, -1, 1)
            env_info = environment.step(actions)[agent.name]
            next_states = env_info.vector_observations
            rewards = np.array(env_info.rewards)
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            episode_rewards.append(rewards)
            episode_scores.append(sum([r for r in rewards])/len(rewards))
            if any(dones):
                break

        scores.append(sum(episode_scores))
        last_100_steps_mean = np.mean(scores[-100:])
        print('\rEpisode {}\tAverage Score: {:.4f}\tLast scores: {}'.format(i_episode,
                                                                               last_100_steps_mean,
                                                                               scores[-5:]), end="")
        if i_episode % 25 == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}\tLast scores: {}'.format(i_episode,
                                                                                   last_100_steps_mean,
                                                                                   scores[-5:]))

        torch.save(agent.actor_local.state_dict(),
                   store_weights_to.replace("eps", str(n_episodes)).replace("role", "actor"))
        torch.save(agent.critic_local.state_dict(),
                   store_weights_to.replace("eps", str(n_episodes)).replace("role", "critic"))
    return scores


def prepare_environment():
    return UnityEnvironment(file_name=ENVIRONMENT_BINARY)


def infer_environment_properties(environment):
    brain_name = environment.brain_names[0]
    brain = environment.brains[brain_name]
    action_size = brain.vector_action_space_size
    env_info = environment.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    states = env_info.vector_observations
    state_size = states.shape[1]
    return brain_name, num_agents, action_size, state_size


def prepare_agent(agent_config: ac_parm, seed):
    return Agent(agent_config, device, seed)


algorithm_factories = {
    "ddpg": prepare_agent
}

simulation_hyperparameter_reference = {
    1:   ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    1, False, False, True, 700, 1),
    2:   ac_parm(-1, -1, int(1e5), "", 256,  0.99, 0.99, 1e-4, 1e-3, 0,    1, False, False, True, 200, 1),
    3:   ac_parm(-1, -1, int(1e5), "", 256,  0.99, 0.99, 1e-4, 1e-3, 0,    1, False, False, True, 700, 1),
    4:   ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    1, False, False, True, 200, 1),
    5:   ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    1, True,  False, False, 0,  1),
    6:   ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False, True, 700, 1),
    7:   ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-3, 1e-3, 0,    1, False, False, True, 700, 1),
    8:   ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False, True, 200, 1),
    9:   ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-3, 1e-3, 0,    1, False, False, True, 200, 1),
    10:  ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, True,  False, False, 0,  1),
    11:  ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-3, 1e-3, 0,    1, True,  False, False, 0,  1),
}


def run_training_session(agent_factory, agent_config: ac_parm, id):
    env = prepare_environment()
    (brain_name, num_agents, action_size, state_size) = infer_environment_properties(env)
    agent_config = agent_config._replace(action_size=action_size, state_size=state_size, name=brain_name)

    scores = []
    for seed in range(agent_config.times):
        agent = agent_factory(agent_config, seed)
        scores.append(
            train(agent, 
                env, 
                random_action_phase_enabled=agent_config.bootstrapping_enabled, 
                random_episodes_number=agent_config.bootstrapping_episodes, 
                store_weights_to=f"{path_prefix}set{id}_weights_role_episode_eps_seed_{seed}.pth"
            ))
    env.close()
    return scores


def ensure_training_run(id: int, parm: ac_parm):
    if os.path.isfile(f"{path_prefix}set{id}_results.json"):
        logging.info(f"Skipping configuration {id} with following parameters {parm}")
    else:
        logging.info(f"Running {id} with following parameters {parm}")
        run_result = run_training_session(algorithm_factories["ddpg"], parm, id)
        with open(f"{path_prefix}set{id}_results.json", "w") as fp:
            json.dump(run_result, fp)


if __name__ == "__main__":
    for parm_id in simulation_hyperparameter_reference:
        ensure_training_run(parm_id, simulation_hyperparameter_reference[parm_id])
