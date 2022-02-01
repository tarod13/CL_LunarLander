import numpy as np

import gym
from agent_1l import create_first_level_agent as create_agent
from buffers import ExperienceBuffer, Task, PixelExperienceSecondLevelMT
from trainers import Second_Level_Trainer_v2 as Trainer
from carl.envs import CARLLunarLanderEnv_defaults as default_params
from carl.envs import CARLLunarLanderEnv
from utils import load_env_model_pairs

import argparse
import pickle
from utils import time_stamp
import itertools
import os


LOAD_PATH = '/home/researcher/Diego/CL_LunarLander/saved_models/'
SAVE_PATH = '/home/researcher/Diego/CL_LunarLander/saved_data/'

DEFAULT_AVERAGE_STEPS = 400
DEFAULT_BUFFER_SIZE = 1000000
DEFAULT_FILE = 'models_to_load_l1.yaml'
DEFAULT_HIDDEN_DIM = 64
DEFAULT_LOAD_BEST = True
DEFAULT_MAX_STEPS = 1000
DEFAULT_N_HEADS = 2
DEFAULT_N_PARTS = 40
DEFAULT_SAVE_STEP_EACH = 1
DEFAULT_USE_SAC_BASELINES = False


def load_agent(
    agent_path, 
    model_id, 
    n_actions=4,
    s_dim=8,
    hidden_dim=64,
    hidden_dim_2=32,
    dueling_layers=2,
    n_heads=2,
    noisy_q_nets=False,
    load_best=True 
    ):
    
    agent = create_agent(
        n_actions = n_actions, 
        feature_dim = s_dim, 
        hidden_dim = hidden_dim,
        hidden_dim_2 = hidden_dim_2,
        dueling_layers = dueling_layers,
        n_heads = n_heads,
        noisy_q_nets = noisy_q_nets
        )

    if load_best:
        agent.load(agent_path + '/best_', model_id)
    else:
        agent.load(agent_path + '/last_', model_id)
    return agent


def store_database(database, n_parts):
    part_size = len(database.buffer) // n_parts
    DB_ID = time_stamp()

    os.makedirs(SAVE_PATH + DB_ID)

    for i in range(0, n_parts):
        PATH = SAVE_PATH + DB_ID + '/DuelingDDQN_training_level1_database_part_' + str(i) + '.p'

        if (i+1) < n_parts:
            pickle.dump(list(itertools.islice(database.buffer, part_size*i, part_size*(i+1))), open(PATH, 'wb'))
        else:
            pickle.dump(list(itertools.islice(database.buffer, part_size*i, None)), open(PATH, 'wb'))


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--average_steps", default=DEFAULT_AVERAGE_STEPS, help="Number of steps taken in each episode (average), default=" + str(DEFAULT_AVERAGE_STEPS))
    parser.add_argument("--buffer_size", default=DEFAULT_BUFFER_SIZE, help="Size of replay buffer, default=" + str(DEFAULT_BUFFER_SIZE))
    parser.add_argument("--cpu", action="store_true", help="Disable cuda")
    parser.add_argument("--file", default=DEFAULT_FILE, help="Name of the folder wih the model info. needed to load them, default=" + DEFAULT_FILE)
    parser.add_argument("--hidden_dim", default=DEFAULT_HIDDEN_DIM, help="Hidden dim in actor and critics, default=" + str(DEFAULT_HIDDEN_DIM))
    parser.add_argument("--load_best", default=DEFAULT_LOAD_BEST, help="If False, the last, instead of the best, model will be loaded")
    parser.add_argument("--max_steps", default=DEFAULT_MAX_STEPS, help="Max number of steps taken in each episode, default=" + str(DEFAULT_MAX_STEPS))
    parser.add_argument("--n_heads", default=DEFAULT_N_HEADS, help="Number of heads in the critic, default=" + str(DEFAULT_N_HEADS))
    parser.add_argument("--n_parts", default=DEFAULT_N_PARTS, help="Number of parts in which the database is divided and store, default=" + str(DEFAULT_N_PARTS))
    parser.add_argument("--render", action="store_true", help="Display agent-env interaction")
    parser.add_argument("--save_step_each", default=DEFAULT_SAVE_STEP_EACH, help="Number of steps to store 1 step in the replay buffer, default=" + str(DEFAULT_SAVE_STEP_EACH))
    args = parser.parse_args()
    
    database = ExperienceBuffer(args.buffer_size, level=3)
    trainer = Trainer()

    env_model_pairs = load_env_model_pairs(args.file)
    n_envs = len(env_model_pairs)
    n_episodes = (args.buffer_size * args.save_step_each) // args.average_steps
    store_video = False

    for env_number, (env_name, info) in enumerate(env_model_pairs.items()):
        task_database = ExperienceBuffer(args.buffer_size//n_envs, level=2)

        env_params = default_params.copy()
        env_params["GRAVITY_X"] = info['Gx']   # TODO: add bounds
        env_params["GRAVITY_Y"] = info['Gy']
        env_params["INITIAL_RANDOM"] = info['RI']
        contexts = {0: env_params}
        env = CARLLunarLanderEnv(contexts=contexts, hide_context=True)
        env_name = (
            f'CARLLunarLander-'
            +f"Gx:{info['Gx']}-"
            +f"Gy:{info['Gy']}-"
            +f"IR:{info['RI']}"
        )
        agent_path = LOAD_PATH + env_name

        model_id = info['id']
        agent = load_agent(
            agent_path, model_id, load_best=args.load_best
        )
    
        print("Populating database of Env " + env_name)
        returns = trainer.loop(
            env, agent, task_database, n_episodes=n_episodes//n_envs, train=False, 
            max_episode_steps=args.max_steps, save_model=False, MODEL_PATH=agent_path, 
            eval_greedy=True, eval_each=20, save_step_each=args.save_step_each,
            use_actor=False
        )
        G = returns.mean()    
        print("Env: " + env_name + ", Mean episode return: {:.2f}".format(G))

        for experience in task_database.buffer:
            task = Task(task=env_number)
            experience_with_task_info = PixelExperienceSecondLevelMT(*experience, *task)
            database.append(experience_with_task_info) 
    
    store_database(database, args.n_parts)
    print("Database stored succesfully")
