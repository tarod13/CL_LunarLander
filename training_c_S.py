import numpy as np

import gym
from agent_c_S import Conceptual_Agent
from concept_optimizers import S_ConceptOptimizer, trajectory_ConceptOptimizer_v2
from buffers import ExperienceBuffer
from trainers import Second_Level_Trainer_v2 as Trainer
from utils import load_database, separate_database, load_policy_PA_ST

import argparse
import pickle
import os
import wandb
import pandas as pd
from tqdm import tqdm

LOAD_PATH = '/home/researcher/Diego/CL_LunarLander/saved_data/'
SAVE_PATH = '/home/researcher/Diego/CL_LunarLander/saved_models/'

DEFAULT_BUFFER_SIZE = 1000000
DEFAULT_BATCH_SIZE = 2048
DEFAULT_BETA_REGULARIZATION = 0
DEFAULT_ETA_REGULARIZATION = 0
DEFAULT_CONSIDER_TASK = False
DEFAULT_DB_ID = '2022-02-01_13-21-14' 
DEFAULT_ID = None 
DEFAULT_STATE_DIM = 8
DEFAULT_LR = 1e-4
DEFAULT_N_PARTS = 40
DEFAULT_N_STEPS = 100000
DEFAULT_N_TASKS = 3
DEFAULT_N_CONCEPTS = 10
DEFAULT_N_ACTIONS = 4
DEFAULT_N_BATCHES = 2
DEFAULT_N_SAVES = 100
DEFAULT_NOISY = False
DEFAULT_UPDATE_RATE = 2e-1
DEFAULT_LATENT_DIM = 64
DEFAULT_DETACH_LOGS = True
DEFAULT_LOAD_POLICY = False
DEFAULT_POLICY_ARTIFACT = 'run-ylkxm56j-PASTfar:v0'
DEFAULT_RESET_EACH = 10000000


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", default=DEFAULT_BETA_REGULARIZATION, help="Regularization level, default=" + str(DEFAULT_BETA_REGULARIZATION))
    parser.add_argument("--eta", default=DEFAULT_ETA_REGULARIZATION, help="Regularization level, default=" + str(DEFAULT_ETA_REGULARIZATION))
    parser.add_argument("--cpu", action="store_true", help="Disable cuda")
    parser.add_argument("--buffer_size", default=DEFAULT_BUFFER_SIZE, help="Size of replay buffer, default=" + str(DEFAULT_BUFFER_SIZE))
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, help="Size of batch used in SGD, default=" + str(DEFAULT_BATCH_SIZE))
    parser.add_argument("--consider_task", default=DEFAULT_CONSIDER_TASK, help="Consider or not task in metric, default=" + str(DEFAULT_CONSIDER_TASK))
    parser.add_argument("--db_id", default=DEFAULT_DB_ID, help="Database ID")
    parser.add_argument("--detach_logs", default=DEFAULT_DETACH_LOGS, help="Pass gradients or not through logarithms, default=" + str(DEFAULT_DETACH_LOGS))
    parser.add_argument("--id", default=DEFAULT_ID, help="ID of model to load")
    parser.add_argument("--latent_dim", default=DEFAULT_LATENT_DIM, help="Concept classifier hidden dim, default=" + str(DEFAULT_LATENT_DIM))
    parser.add_argument("--load_policy", default=DEFAULT_LOAD_POLICY, help="Use prior policy from concepts to actions, default=" + str(DEFAULT_LOAD_POLICY))
    parser.add_argument("--lr", default=DEFAULT_LR, help="Learning rate, default=" + str(DEFAULT_LR))
    parser.add_argument("--n_parts", default=DEFAULT_N_PARTS, help="Number of parts in which the database is divided and store, default=" + str(DEFAULT_N_PARTS))
    parser.add_argument("--n_saves", default=DEFAULT_N_SAVES, help="Number of times the model is saved, default=" + str(DEFAULT_N_SAVES))
    parser.add_argument("--noisy", default=DEFAULT_NOISY, help="Use noisy layers in the concept module")
    parser.add_argument("--n_steps", default=DEFAULT_N_STEPS, help="Number of SGD steps taken, default=" + str(DEFAULT_N_STEPS))
    parser.add_argument("--n_tasks", default=DEFAULT_N_TASKS, help="Number of tasks, default=" + str(DEFAULT_N_TASKS))
    parser.add_argument("--n_actions", default=DEFAULT_N_ACTIONS, help="Number of actions, default=" + str(DEFAULT_N_ACTIONS))
    parser.add_argument("--n_batches", default=DEFAULT_N_BATCHES, type=int, help="Number of batches for estimation, default=" + str(DEFAULT_N_BATCHES))
    parser.add_argument("--n_concepts", default=DEFAULT_N_CONCEPTS, help="Number of concepts, default=" + str(DEFAULT_N_CONCEPTS))
    parser.add_argument("--update_rate", default=DEFAULT_UPDATE_RATE, help="Update rate for joint probability estimation, default=" + str(DEFAULT_UPDATE_RATE))
    parser.add_argument("--policy_artifact", default=DEFAULT_POLICY_ARTIFACT, help="Weights and Biases artifact where prior policy is stored")
    parser.add_argument("--reset_each", default=DEFAULT_RESET_EACH, help="Number of iterations before resetting joint distribution estimate")
    parser.add_argument("--state_dim", default=DEFAULT_STATE_DIM, help="Dimensionality of inner state, default=" + str(DEFAULT_STATE_DIM))
    args = parser.parse_args()

    project_name = 'SAC_box2d_conceptual_level'

    # Initilize Weights-and-Biases project
    wandb_run = wandb.init(project=project_name)

    # Log hyperparameters in WandB project
    wandb.config.update(args)

    device = 'cuda' if not args.cpu else 'cpu'
    
    # Load dataset and separate in train and test sets
    database = load_database(
        args.n_parts, LOAD_PATH, args.db_id, args.buffer_size, 3, 'DuelingDDQN'
    )
    train_database, test_database = separate_database(database)

    # Create or load agent
    conceptual_agent = Conceptual_Agent(
        args.state_dim, args.latent_dim, 
        args.n_concepts, args.noisy, args.lr
    ).to(device)
    if args.id is not None:
        conceptual_agent.load(SAVE_PATH, args.id)

    # Select concept optimizer
    if args.load_policy:
        policy_from_concepts_to_actions = load_policy_PA_ST(
            args.policy_artifact, wandb_run)

        concept_optimizer = trajectory_ConceptOptimizer_v2(
            policy_from_concepts_to_actions, args.batch_size, 
            args.beta, args.eta, args.n_batches, args.update_rate
        )
    else:
        concept_optimizer = S_ConceptOptimizer(
            args.batch_size, args.beta, args.eta, args.n_batches,  
            args.update_rate, args.consider_task, args.detach_logs
        )

    os.makedirs(SAVE_PATH, exist_ok=True)

    # Train concept classifier
    for step in tqdm(range(0, args.n_steps)):
        initialization = (step % args.reset_each) == 0
        train_metrics = concept_optimizer.optimize(
            conceptual_agent, train_database, args.n_actions, 
            args.n_tasks, initialization, train=True
        )        
        test_metrics = concept_optimizer.optimize(
            conceptual_agent, test_database, args.n_actions, 
            args.n_tasks, initialization, train=False
        )
        
        # Export metrics
        metrics = {**train_metrics, **test_metrics}
        metrics['step'] = step

        NT, NS, NA = concept_optimizer.PAST.shape
        TT = np.array(list(range(NT)))
        SS = np.array(list(range(NS)))      
        AA = np.array(list(range(NA)))
        II = np.meshgrid(TT,SS,AA, indexing='ij')
        II.append(concept_optimizer.PAST.detach().cpu().numpy())
        data_ = [x.reshape(-1,1) for x in II]
        data_np = np.concatenate(data_, axis=1)
        data_table = wandb.Table(columns=['T','S','A','P(A,S,T)'], data=data_np)

        metrics['P(AST)'] = data_table
        wandb.log(metrics)
    
        # Save concept classifier module
        should_save = ((step+1) % (args.n_steps // args.n_saves)) == 0
        if should_save:
            v = (step+1) // (args.n_steps // args.n_saves)
            conceptual_agent.save(SAVE_PATH, v=v)
