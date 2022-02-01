import numpy as np

import gym


from agent_1l_discrete import create_second_level_agent as create_agent
from buffers import ExperienceBuffer as Buffer
from trainers import Second_Level_Trainer as Trainer
from carl.envs import CARLLunarLanderEnv_defaults as default_params
from carl.envs import CARLLunarLanderEnv

import wandb
import argparse
import os
import collections

os.environ["WANDB_START_METHOD"] = "thread"

DEFAULT_GRAVITY_X = 0
DEFAULT_GRAVITY_Y = -10
DEFAULT_ENV_NAME = (
    f'CARLLunarLander-'
    +f'Gx:{DEFAULT_GRAVITY_X}-'
    +f'Gy:{DEFAULT_GRAVITY_Y}'
)
DEFAULT_N_STEPS_IN_SECOND_LEVEL_EPISODE = 1000
DEFAULT_ACTOR_LOSS = 'mse'
DEFAULT_BUFFER_SIZE = 1000000
DEFAULT_N_EPISODES = 20000
DEFAULT_ID = '2001-01-15_19-10-56'
DEFAULT_CLIP_Q_ERROR = False
DEFAULT_CLIP_VALUE_Q_ERROR = 5.0
DEFAULT_CLIP_VALUE = 1.0
DEFAULT_CLIP_ACTOR_DIVERGENCE = False
DEFAULT_CLIP_ACTOR_VALUE = 1.0
DEFAULT_DISCOUNT_FACTOR = 0.99
DEFAULT_BATCH_SIZE = 64
DEFAULT_MIN_EPSILON = 0.01
DEFAULT_INIT_EPSILON = 1.0
DEFAULT_DELTA_EPSILON = 0
DEFAULT_DECAY_EPSILON = 0.995
DEFAULT_ENTROPY_FACTOR = 0.1
DEFAULT_ENTROPY_UPDATE_RATE = 0.005
DEFAULT_WEIGHT_Q_LOSS = 0.5
DEFAULT_INIT_LOG_ALPHA = -20
DEFAULT_LEARN_ALPHA = False
DEFAULT_LR = 1e-3
DEFAULT_LR_ACTOR = 1e-3 
DEFAULT_LR_ALPHA = 1e-3 
DEFAULT_ALPHA_V_WEIGHT = 0
DEFAULT_INITIALIZE_SAC = False
DEFAULT_INITIALIZATION_STEPS_SAC = 20000
DEFAULT_NOISY_ACTOR_CRITIC = False
DEFAULT_SAVE_STEP_EACH = 1
DEFAULT_TRAIN_EACH = 1
DEFAULT_N_STEP_TD = 1
DEFAULT_PARALLEL_Q_NETS = True
DEFAULT_N_HEADS = 2
DEFAULT_HIDDEN_DIM = 64
DEFAULT_N_AGENTS = 1
DEFAULT_NORMALIZE_Q_ERROR = True
DEFAULT_RESTRAIN_POLICY_UPDATE = False
DEFAULT_USE_H_MEAN = False
DEFAULT_USE_ENTROPY = True
DEFAULT_NORMALIZE_Q_DIST = False #True
DEFAULT_TARGET_UPDATE_RATE = 5e-2
DEFAULT_EVAL_GREEDY = True
DEFAULT_USE_PIXELS = False
DEFAULT_USE_ACTOR = False
DEFAULT_WRAPPER_SCALE = 0.25
DEFAULT_REWARD_SCALE = 0.1

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Disable cuda")
    parser.add_argument("--render", action="store_true", help="Display agent-env interaction")
    parser.add_argument("--eval", action="store_true", help="Train (False) or evaluate (True) the agent")
    parser.add_argument("--gx", default=DEFAULT_GRAVITY_X, help=f"Gravity in x-asis, default={DEFAULT_GRAVITY_X}")
    parser.add_argument("--gy", default=DEFAULT_GRAVITY_Y, help=f"Gravity in y-asis, default={DEFAULT_GRAVITY_Y}")
    parser.add_argument("--env_name", default=DEFAULT_ENV_NAME, help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--n_steps_in_second_level_episode", default=DEFAULT_N_STEPS_IN_SECOND_LEVEL_EPISODE, help="Number of second decision" +
        "level steps taken in each episode, default=" + str(DEFAULT_N_STEPS_IN_SECOND_LEVEL_EPISODE))
    parser.add_argument("--actor_loss", default=DEFAULT_ACTOR_LOSS, help="Function used to train the actor")
    parser.add_argument("--buffer_size", default=DEFAULT_BUFFER_SIZE, help="Size of replay buffer, default=" + str(DEFAULT_BUFFER_SIZE))
    parser.add_argument("--n_episodes", default=DEFAULT_N_EPISODES, type=int, help="Number of episodes, default=" + str(DEFAULT_N_EPISODES))
    parser.add_argument("--discount_factor", default=DEFAULT_DISCOUNT_FACTOR, help="Discount factor (0,1), default=" + str(DEFAULT_DISCOUNT_FACTOR))
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, help="Batch size, default=" + str(DEFAULT_BATCH_SIZE))
    parser.add_argument("--clip_q_error", default=DEFAULT_CLIP_Q_ERROR, help="Clip q error wrt targets, default=" + str(DEFAULT_CLIP_Q_ERROR))
    parser.add_argument("--clip_value_q_error", default=DEFAULT_CLIP_VALUE_Q_ERROR, help="Clip value when clipping q error wrt targets, default=" + str(DEFAULT_CLIP_VALUE_Q_ERROR))
    parser.add_argument("--init_epsilon", default=DEFAULT_INIT_EPSILON, help="Initial annealing factor for entropy, default=" + str(DEFAULT_INIT_EPSILON))
    parser.add_argument("--min_epsilon", default=DEFAULT_MIN_EPSILON, help="Minimum annealing factor for entropy, default=" + str(DEFAULT_MIN_EPSILON))
    parser.add_argument("--delta_epsilon", default=DEFAULT_DELTA_EPSILON, help="Decreasing rate of annealing factor for entropy, default=" + str(DEFAULT_DELTA_EPSILON))
    parser.add_argument("--decay_epsilon", default=DEFAULT_DECAY_EPSILON, help="Decreasing factor of exploration rate, default=" + str(DEFAULT_DECAY_EPSILON))
    parser.add_argument("--entropy_factor", default=DEFAULT_ENTROPY_FACTOR, help="Entropy coefficient, default=" + str(DEFAULT_ENTROPY_FACTOR))
    parser.add_argument("--entropy_update_rate", default=DEFAULT_ENTROPY_UPDATE_RATE, help="Mean entropy update rate, default=" + str(DEFAULT_ENTROPY_UPDATE_RATE))
    parser.add_argument("--weight_q_loss", default=DEFAULT_WEIGHT_Q_LOSS, help="Weight of critics' loss, default=" + str(DEFAULT_WEIGHT_Q_LOSS))
    parser.add_argument("--init_log_alpha", default=DEFAULT_INIT_LOG_ALPHA, help="Initial temperature parameter, default=" + str(DEFAULT_INIT_LOG_ALPHA))
    parser.add_argument("--learn_alpha", default=DEFAULT_LEARN_ALPHA, help="Wether to learn the temperature parameter, default=" + str(DEFAULT_LEARN_ALPHA))
    parser.add_argument("--lr", default=DEFAULT_LR, help="Learning rate, default=" + str(DEFAULT_LR))
    parser.add_argument("--lr_actor", default=DEFAULT_LR_ACTOR, help="Learning rate for actor, default=" + str(DEFAULT_LR_ACTOR))
    parser.add_argument("--lr_alpha", default=DEFAULT_LR_ALPHA, help="Learning rate for temperature, default=" + str(DEFAULT_LR_ALPHA))
    parser.add_argument("--alpha_v_weight", default=DEFAULT_ALPHA_V_WEIGHT, help="Weight for entropy velocity in temperature loss, default=" + str(DEFAULT_ALPHA_V_WEIGHT))
    parser.add_argument("--init_sac", default=DEFAULT_INITIALIZE_SAC, help="Initialize the replay buffer of the agent by acting randomly for a specified number of steps ")
    parser.add_argument("--init_steps_sac", default=DEFAULT_INITIALIZATION_STEPS_SAC, help="Minimum observations in replay buffer to start learning, default=" + str(DEFAULT_INITIALIZATION_STEPS_SAC))
    parser.add_argument("--load_id", default=None, help="Model ID to load, default=None")
    parser.add_argument("--load_best", action="store_true", help="If flag is used the best model will be loaded (if ID is provided)")
    parser.add_argument("--eval_greedy", default=DEFAULT_EVAL_GREEDY, help="If true, then the evaluation policy is greedy")
    parser.add_argument("--noisy_ac", default=DEFAULT_NOISY_ACTOR_CRITIC, help="Use noisy layers in the actor-critic module")
    parser.add_argument("--save_step_each", default=DEFAULT_SAVE_STEP_EACH, help="Number of steps to store 1 step in the replay buffer, default=" + str(DEFAULT_SAVE_STEP_EACH))
    parser.add_argument("--train_each", default=DEFAULT_TRAIN_EACH, help="Number of steps ellapsed to train once, default=" + str(DEFAULT_TRAIN_EACH))
    parser.add_argument("--n_step_td", default=DEFAULT_N_STEP_TD, help="Number of steps to calculate temporal differences, default=" + str(DEFAULT_N_STEP_TD))
    parser.add_argument("--parallel_q_nets", default=DEFAULT_PARALLEL_Q_NETS, help="Use or not parallel q nets in actor critic, default=" + str(DEFAULT_PARALLEL_Q_NETS))
    parser.add_argument("--n_heads", default=DEFAULT_N_HEADS, help="Number of heads in the critic, default=" + str(DEFAULT_N_HEADS))
    parser.add_argument("--normalize_q_error", default=DEFAULT_NORMALIZE_Q_ERROR, help="Normalize critic error dividing by maximum, default=" + str(DEFAULT_NORMALIZE_Q_ERROR))
    parser.add_argument("--restrain_pi_update", default=DEFAULT_RESTRAIN_POLICY_UPDATE, help="Penalize policy changes that are too large, default=" + str(DEFAULT_RESTRAIN_POLICY_UPDATE))
    parser.add_argument("--clip_value", default=DEFAULT_CLIP_VALUE, help="Clip value for optimizer")
    parser.add_argument("--clip_actor", default=DEFAULT_CLIP_ACTOR_DIVERGENCE, help="Clip divergence in actor loss")
    parser.add_argument("--clip_actor_value", default=DEFAULT_CLIP_ACTOR_VALUE, help="Limit divergence accepted in actor loss")
    parser.add_argument("--n_agents", default=DEFAULT_N_AGENTS, type=int, help="Number of agents")
    parser.add_argument("--use_H_mean", default=DEFAULT_USE_H_MEAN, help="Wether to use H mean in SAC's critic loss")
    parser.add_argument("--use_entropy", default=DEFAULT_USE_ENTROPY, help="Use SAC with or without entropy")
    parser.add_argument("--normalize_q_dist", default=DEFAULT_NORMALIZE_Q_DIST, help="Normalize q target distribution")
    parser.add_argument("--target_update_rate", default=DEFAULT_TARGET_UPDATE_RATE, help="Update rate for target q networks")
    parser.add_argument("--wrapper_scale", default=DEFAULT_WRAPPER_SCALE, help="Fraction of the original observations in use (0,1]")
    parser.add_argument("--reward_scale", default=DEFAULT_REWARD_SCALE, help="Reward scale")
    parser.add_argument("--use_pixels", default=DEFAULT_USE_PIXELS, help="Use pixels instead of vector state information")
    parser.add_argument("--use_actor", default=DEFAULT_USE_ACTOR, help="Use actor or explore with soft-epsilon policy")
    parser.add_argument("--hidden_dim", default=DEFAULT_HIDDEN_DIM, help="Hidden dim in actor and critics, default=" + str(DEFAULT_HIDDEN_DIM))
    args = parser.parse_args()

    MODEL_PATH = '/home/researcher/Diego/CL_LunarLander/saved_models/'
    project_name = 'SAC_box2d'
    
    # Set hyperparameters
    n_episodes = 1 if args.eval else args.n_episodes
    optimizer_kwargs = {
        'learn_alpha': args.learn_alpha,
        'batch_size': args.batch_size, 
        'discount_factor': args.discount_factor,
        'init_epsilon': args.init_epsilon,
        'min_epsilon': args.min_epsilon,
        'delta_epsilon': args.delta_epsilon,
        'entropy_factor': args.entropy_factor,
        'weight_q_loss': args.weight_q_loss,
        'alpha_v_weight': args.alpha_v_weight,
        'entropy_update_rate': args.entropy_update_rate,
        'clip_value': args.clip_value,
        'restrain_policy_update': args.restrain_pi_update,
        'clip_q_error': args.clip_q_error,
        'clip_value_q_error': args.clip_value_q_error,
        'use_H_mean': args.use_H_mean,
        'use_entropy': args.use_entropy,
        'normalize_q_error': args.normalize_q_error,
        'normalize_q_dist': args.normalize_q_dist,
        'target_update_rate': args.target_update_rate,
        'clip_actor': args.clip_actor,
        'clip_actor_value': args.clip_actor_value,
        'actor_loss_function': args.actor_loss,
        'reward_scale': args.reward_scale,
    }

    store_video = args.eval
    wandb_project = not args.eval

    # Initilize Weights-and-Biases project
    if wandb_project:
        wandb.init(project=project_name)

        # Log hyperparameters in WandB project
        wandb.config.update(args)
        # wandb.config.healthy_reward = DEFAULT_HEALTHY_REWARD 


    env_params = default_params.copy()
    env_params["GRAVITY_X"] = args.gx   # TODO: add bounds
    env_params["GRAVITY_Y"] = args.gy
    contexts = {0: env_params}
    env = CARLLunarLanderEnv(contexts=contexts, hide_context=True)
    # env = gym.make('LunarLander-v2')

    n_actions = env.action_space.n
    optimizer_kwargs['n_actions'] = n_actions
    feature_dim = env.observation_space.shape[0]

    agent = create_agent(
        use_pixels=args.use_pixels, n_actions=n_actions, 
        noisy=args.noisy_ac, n_heads=args.n_heads, 
        init_log_alpha=args.init_log_alpha, 
        feature_dim=feature_dim, parallel=args.parallel_q_nets,
        lr=args.lr, lr_alpha=args.lr_alpha, lr_actor=args.lr_actor,
        int_heads=False, hidden_dim=args.hidden_dim
        )
    
    if args.load_id is not None:
        if args.load_best:
            agent.load(MODEL_PATH + args.env_name + '/best_', args.load_id)
        else:
            agent.load(MODEL_PATH + args.env_name + '/last_', args.load_id)
    agents = collections.deque(maxlen=args.n_agents)
    agents.append(agent)
    
    os.makedirs(MODEL_PATH + args.env_name, exist_ok=True)

    database = Buffer(args.buffer_size, level=2)

    trainer = Trainer(optimizer_kwargs=optimizer_kwargs)
    returns = trainer.loop(
        env, agents, database, n_episodes=n_episodes, render=args.render, 
        max_episode_steps=args.n_steps_in_second_level_episode, 
        store_video=store_video, wandb_project=wandb_project, 
        MODEL_PATH=MODEL_PATH+args.env_name, train=(not args.eval),
        save_step_each=args.save_step_each, train_each=args.train_each, 
        n_step_td=args.n_step_td, greedy_sampling=args.eval_greedy,
        initialization=args.init_sac, init_buffer_size=args.init_steps_sac,
        use_actor=args.use_actor, decay_epsilon=args.decay_epsilon
    )
    G = returns.mean()    
    print("Mean episode return: {:.2f}".format(G)) 