import numpy as np
import gym

from agent_1l import create_first_level_agent as create_agent
from buffers import ExperienceBuffer as Buffer
from trainers import Second_Level_Trainer_v2 as Trainer
from carl.envs import CARLLunarLanderEnv_defaults as default_params
from carl.envs import CARLLunarLanderEnv

import wandb
import argparse
import os

os.environ["WANDB_START_METHOD"] = "thread"

DEFAULT_USE_CARL = True
DEFAULT_GRAVITY_X = 5     #  0 # 4   # 5
DEFAULT_GRAVITY_Y = -5*np.sqrt(3)   #-10 #-9.2 #-5*np.sqrt(3)
DEFAULT_INITIAL_RANDOM = 1000
DEFAULT_ENV_NAME = (
    f'CARLLunarLander-'
    +f'Gx:{DEFAULT_GRAVITY_X}-'
    +f'Gy:{DEFAULT_GRAVITY_Y}-'
    +F'IR:{DEFAULT_INITIAL_RANDOM}'
)

DEFAULT_ACTOR_LOSS = 'mse'
DEFAULT_BATCH_SIZE = 64
DEFAULT_BUFFER_SIZE = 1000000
DEFAULT_CLIP_VALUE = 1.0
DEFAULT_DELTA_EPSILON = 0
DEFAULT_DECAY_EPSILON = 0.995
DEFAULT_DISCOUNT_FACTOR = 0.99
DEFAULT_DUELING_LAYERS = 2
DEFAULT_ENTROPY_UPDATE_RATE = 0.005
DEFAULT_EVAL_GREEDY = True
DEFAULT_HIDDEN_DIM = 64
DEFAULT_HIDDEN_DIM_2 = 32
DEFAULT_INIT_EPSILON = 1.0
DEFAULT_INIT_LOG_ALPHA = 0.0
DEFAULT_INIT_NOISE = 0.25
DEFAULT_LEARN_ALPHA = False
DEFAULT_LOAD_ID = None
DEFAULT_LR = 5e-4
DEFAULT_LR_ACTOR = 5e-4 
DEFAULT_LR_ALPHA = 5e-4 
DEFAULT_MIN_EPSILON = 0.01
DEFAULT_N_EPISODES = 4000
DEFAULT_N_HEADS = 2
DEFAULT_N_STEPS_IN_EPISODE = 1000
DEFAULT_NOISY_Q_NETS = True
DEFAULT_NORMALIZE_Q_ERROR = True
DEFAULT_REWARD_SCALE = 1.0
DEFAULT_SAVE_STEP_EACH = 1
DEFAULT_TARGET_UPDATE_RATE = 1e-3
DEFAULT_TRAIN_EACH = 4
DEFAULT_USE_ACTOR = False
DEFAULT_USE_ENTROPY = False
DEFAULT_USE_GRADIENT_CLIP = True
DEFAULT_USE_H_MEAN = True


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_loss", default=DEFAULT_ACTOR_LOSS, help="Function used to train the actor")
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, help="Batch size, default=" + str(DEFAULT_BATCH_SIZE))
    parser.add_argument("--buffer_size", default=DEFAULT_BUFFER_SIZE, help="Size of replay buffer, default=" + str(DEFAULT_BUFFER_SIZE))
    parser.add_argument("--clip_value", default=DEFAULT_CLIP_VALUE, help="Clip value for optimizer")
    parser.add_argument("--cpu", action="store_true", help="Disable cuda")
    parser.add_argument("--decay_epsilon", default=DEFAULT_DECAY_EPSILON, help="Decreasing factor of exploration rate, default=" + str(DEFAULT_DECAY_EPSILON))
    parser.add_argument("--delta_epsilon", default=DEFAULT_DELTA_EPSILON, help="Decreasing rate of annealing factor for entropy, default=" + str(DEFAULT_DELTA_EPSILON))
    parser.add_argument("--discount_factor", default=DEFAULT_DISCOUNT_FACTOR, help="Discount factor (0,1), default=" + str(DEFAULT_DISCOUNT_FACTOR))
    parser.add_argument("--dueling_layers", default=DEFAULT_DUELING_LAYERS, help="Number of layers used in dueling architecture")
    parser.add_argument("--entropy_update_rate", default=DEFAULT_ENTROPY_UPDATE_RATE, help="Mean entropy update rate, default=" + str(DEFAULT_ENTROPY_UPDATE_RATE))
    parser.add_argument("--env_name", default=DEFAULT_ENV_NAME, help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--eval", action="store_true", help="Train (False) or evaluate (True) the agent")
    parser.add_argument("--eval_greedy", default=DEFAULT_EVAL_GREEDY, help="If true, then the evaluation policy is greedy")
    parser.add_argument("--gx", default=DEFAULT_GRAVITY_X, help=f"Gravity in x-asis, default={DEFAULT_GRAVITY_X}")
    parser.add_argument("--gy", default=DEFAULT_GRAVITY_Y, help=f"Gravity in y-asis, default={DEFAULT_GRAVITY_Y}")
    parser.add_argument("--hidden_dim", default=DEFAULT_HIDDEN_DIM, help="Hidden dim in actor and critics, default=" + str(DEFAULT_HIDDEN_DIM))
    parser.add_argument("--hidden_dim_2", default=DEFAULT_HIDDEN_DIM_2, help="Hidden dim 2 in dueling critics, default=" + str(DEFAULT_HIDDEN_DIM_2))
    parser.add_argument("--init_epsilon", default=DEFAULT_INIT_EPSILON, help="Initial annealing factor for entropy, default=" + str(DEFAULT_INIT_EPSILON))
    parser.add_argument("--init_log_alpha", default=DEFAULT_INIT_LOG_ALPHA, help="Initial temperature parameter, default=" + str(DEFAULT_INIT_LOG_ALPHA))
    parser.add_argument("--init_noise", default=DEFAULT_INIT_NOISE, help="Initial noise level in noisy layers, default=" + str(DEFAULT_INIT_NOISE))
    parser.add_argument("--initial_random", default=DEFAULT_INITIAL_RANDOM, help="LunarLander initial random parameter, default=" + str(DEFAULT_INITIAL_RANDOM))
    parser.add_argument("--learn_alpha", default=DEFAULT_LEARN_ALPHA, help="Wether to learn the temperature parameter, default=" + str(DEFAULT_LEARN_ALPHA))
    parser.add_argument("--lr", default=DEFAULT_LR, help="Learning rate, default=" + str(DEFAULT_LR))
    parser.add_argument("--lr_actor", default=DEFAULT_LR_ACTOR, help="Learning rate for actor, default=" + str(DEFAULT_LR_ACTOR))
    parser.add_argument("--lr_alpha", default=DEFAULT_LR_ALPHA, help="Learning rate for temperature, default=" + str(DEFAULT_LR_ALPHA))
    parser.add_argument("--load_id", default=DEFAULT_LOAD_ID, help="Model ID to load, default=None")
    parser.add_argument("--load_best", action="store_true", help="If flag is used the best model will be loaded (if ID is provided)")
    parser.add_argument("--min_epsilon", default=DEFAULT_MIN_EPSILON, help="Minimum annealing factor for entropy, default=" + str(DEFAULT_MIN_EPSILON))
    parser.add_argument("--n_episodes", default=DEFAULT_N_EPISODES, type=int, help="Number of episodes, default=" + str(DEFAULT_N_EPISODES))
    parser.add_argument("--n_heads", default=DEFAULT_N_HEADS, help="Number of heads in the critic, default=" + str(DEFAULT_N_HEADS))
    parser.add_argument("--n_steps_episode", default=DEFAULT_N_STEPS_IN_EPISODE, help="Number of steps taken in each episode, default=" + str(DEFAULT_N_STEPS_IN_EPISODE))
    parser.add_argument("--noisy_q_nets", default=DEFAULT_NOISY_Q_NETS, help="Use noisy layers in Q nets, default=" + str(DEFAULT_NOISY_Q_NETS))
    parser.add_argument("--normalize_q_error", default=DEFAULT_NORMALIZE_Q_ERROR, help="Normalize critic error dividing by maximum, default=" + str(DEFAULT_NORMALIZE_Q_ERROR))
    parser.add_argument("--render", action="store_true", help="Display agent-env interaction")
    parser.add_argument("--reward_scale", default=DEFAULT_REWARD_SCALE, help="Reward scale")
    parser.add_argument("--save_step_each", default=DEFAULT_SAVE_STEP_EACH, help="Number of steps to store 1 step in the replay buffer, default=" + str(DEFAULT_SAVE_STEP_EACH))
    parser.add_argument("--target_update_rate", default=DEFAULT_TARGET_UPDATE_RATE, help="Update rate for target q networks")
    parser.add_argument("--train_each", default=DEFAULT_TRAIN_EACH, help="Number of steps ellapsed to train once, default=" + str(DEFAULT_TRAIN_EACH))
    parser.add_argument("--use_actor", default=DEFAULT_USE_ACTOR, help="Use actor or explore with soft-epsilon policy")
    parser.add_argument("--use_carl", default=DEFAULT_USE_CARL, help="Use OpenAI env version or CARL version")
    parser.add_argument("--use_entropy", default=DEFAULT_USE_ENTROPY, help="Use SAC with or without entropy")
    parser.add_argument("--use_gradient_clip", default=DEFAULT_USE_GRADIENT_CLIP, help="Use gradient clip")
    parser.add_argument("--use_H_mean", default=DEFAULT_USE_H_MEAN, help="Wether to use H mean in SAC's critic loss")
    args = parser.parse_args()

    if args.noisy_q_nets:
        args.init_epsilon = 0.0
        args.min_epsilon = 0.0
    elif args.use_actor:
        args.min_epsilon = 0.2

    MODEL_PATH = '/home/researcher/Diego/CL_LunarLander/saved_models/'
    project_name = 'SAC_box2d'
    
    # Set hyperparameters
    n_episodes = 1 if args.eval else args.n_episodes
    optimizer_kwargs = {
        'actor_loss_function': args.actor_loss,
        'batch_size': args.batch_size, 
        'clip_value': args.clip_value,
        'delta_epsilon': args.delta_epsilon,
        'discount_factor': args.discount_factor,
        'entropy_update_rate': args.entropy_update_rate,
        'init_epsilon': args.init_epsilon,
        'learn_alpha': args.learn_alpha,
        'min_epsilon': args.min_epsilon,
        'normalize_q_error': args.normalize_q_error,
        'reward_scale': args.reward_scale,
        'target_update_rate': args.target_update_rate,
        'use_actor': args.use_actor,
        'use_entropy': args.use_entropy,
        'use_gradient_clip': args.use_gradient_clip,
        'use_H_mean': args.use_H_mean,
    }

    store_video = args.eval
    wandb_project = not args.eval

    # Initilize Weights-and-Biases project
    if wandb_project:
        wandb.init(project=project_name)

        # Log hyperparameters in WandB project
        wandb.config.update(args)
        # wandb.config.healthy_reward = DEFAULT_HEALTHY_REWARD 


    if args.use_carl:
        env_params = default_params.copy()
        env_params["GRAVITY_X"] = args.gx   # TODO: add bounds
        env_params["GRAVITY_Y"] = args.gy
        env_params["INITIAL_RANDOM"] = args.initial_random
        contexts = {0: env_params}
        env = CARLLunarLanderEnv(contexts=contexts, hide_context=True)
    else:
        env = gym.make('LunarLander-v2')

    n_actions = env.action_space.n
    optimizer_kwargs['n_actions'] = n_actions
    feature_dim = env.observation_space.shape[0]

    agent = create_agent(
        n_actions = n_actions, 
        init_log_alpha = args.init_log_alpha, 
        feature_dim = feature_dim, 
        lr = args.lr, 
        lr_alpha = args.lr_alpha, 
        lr_actor = args.lr_actor,
        hidden_dim = args.hidden_dim,
        hidden_dim_2 = args.hidden_dim_2,
        dueling_layers = args.dueling_layers,
        n_heads = args.n_heads,
        noisy_q_nets = args.noisy_q_nets,
        init_noise = args.init_noise
        )
    wandb.config.agent_id = agent._id
    
    if args.load_id is not None:
        if args.load_best:
            agent.load(MODEL_PATH + args.env_name + '/best_', args.load_id)
        else:
            agent.load(MODEL_PATH + args.env_name + '/last_', args.load_id)
    
    os.makedirs(MODEL_PATH + args.env_name, exist_ok=True)

    database = Buffer(args.buffer_size, level=2)

    trainer = Trainer(optimizer_kwargs=optimizer_kwargs)
    returns = trainer.loop(
        env, agent, database, n_episodes=n_episodes, render=args.render, 
        max_episode_steps=args.n_steps_episode, 
        store_video=store_video, wandb_project=wandb_project, 
        MODEL_PATH=MODEL_PATH+args.env_name, train=(not args.eval),
        save_step_each=args.save_step_each, train_each=args.train_each, 
        eval_greedy=args.eval_greedy, use_actor=args.use_actor, 
        decay_epsilon=args.decay_epsilon
    )
    G = returns.mean()    
    print("Mean episode return: {:.2f}".format(G)) 