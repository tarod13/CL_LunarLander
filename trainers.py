import collections
import numpy as np
from buffers import (
    ExperienceBuffer,
    PixelExperienceSecondLevel, PixelExperienceThirdLevel
    )
from policy_optimizers import (
    Second_Level_SAC_PolicyOptimizer_v2,
    Third_Level_SAC_PolicyOptimizer
)

import wandb
from tqdm import tqdm

import cv2
video_folder = '/home/researcher/Diego/CL_LunarLander/videos/'


class Second_Level_Trainer_v2:
    def __init__(self, optimizer_kwargs={}):
        self.optimizer = Second_Level_SAC_PolicyOptimizer_v2(**optimizer_kwargs)
        
    def loop(
        self, 
        env, 
        agent, 
        database, 
        n_episodes=10, 
        train=True,
        max_episode_steps=2000, 
        train_each=1, 
        update_database=True, 
        render=False, 
        store_video=False, 
        wandb_project=False, 
        save_model=True, 
        save_model_each=50, 
        MODEL_PATH='', 
        save_step_each=2, 
        eval_greedy=False,
        eval_each=5, 
        use_actor=True,
        decay_epsilon=1.0
        ):

        best_return = -np.infty
        explore = train or (not eval_greedy)

        if store_video:
            video_filename = (
                video_folder
                + 'CARLLunarLander_'+str(agent.get_id())
                + '.mp4'
            )
            fourcc = cv2.VideoWriter_fourcc(*'avc1')#*'avc1'
            video = cv2.VideoWriter(video_filename, fourcc, 50, (600, 400))

        returns = []
        disable_tqdm = (not train) and (n_episodes==1)
        for episode in tqdm(range(0, n_episodes), disable=disable_tqdm):
            step_counter = 0
            episode_done = False
            episode_return = 0.0

            state = env.reset()

            while not episode_done:
                action, dist = agent.sample_action(
                    state, 
                    explore=explore, 
                    use_actor=use_actor, 
                    eps=self.optimizer.epsilon
                )
                
                if render:
                    env.render()

                next_state, reward, done, info = env.step(action)

                if store_video:
                    img = env.render('rgb_array')
                    video.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                if update_database:
                    step = PixelExperienceSecondLevel(
                        state, action, reward,
                        done, next_state
                    )

                    should_save_step = (step_counter % save_step_each) == 0
                    if should_save_step:
                        database.append(step)

                episode_return += reward
                state = next_state.copy()
                
                step_counter += 1

                should_train_in_this_step = train and ((step_counter % train_each) == 0)
                if should_train_in_this_step:
                    metrics = self.optimizer.optimize(agent, database)
                    if wandb_project and metrics is not None:
                        metrics['step'] = step_counter
                        wandb.log(metrics)

                if step_counter >= max_episode_steps or done:
                    episode_done = True

            # Update exploration factor
            self.optimizer.epsilon = np.max([
                self.optimizer.epsilon * decay_epsilon, 
                self.optimizer.min_epsilon
            ])

            # Log return
            returns.append(episode_return)
            if wandb_project and train:
                wandb.log({'episode': episode, 'return': episode_return})

            # Save model
            if save_model and ((episode + 1) % save_model_each == 0):
                agent.save(MODEL_PATH + '/')
            
            if train and (episode_return > best_return):
                best_return = episode_return
                agent.save(MODEL_PATH + '/', best=True)
            
            # Eval agent
            if train and ((episode+1) % eval_each == 0):
                eval_returns = self.loop(env, agent, None, n_episodes=1, train=False, 
                    max_episode_steps=max_episode_steps, update_database=False,
                    render=False, store_video=True, wandb_project=wandb_project,
                    save_model=False, eval_greedy=eval_greedy,
                    use_actor=use_actor
                )
                wandb.log(
                    {
                        'episode_eval': episode//eval_each, 
                        'eval_return': eval_returns.mean(),
                    }
                ) 

        return_array = np.array(returns)

        if store_video:
            video.release()
            wandb.log(
                {'video': wandb.Video(video_filename, fps=4, format='mp4')}
            )

        if render:
            env.close()

        return return_array 


class Third_Level_Trainer:
    def __init__(self, optimizer_kwargs={}):
        self.optimizer = Third_Level_SAC_PolicyOptimizer(**optimizer_kwargs)
        
    def loop(self, env, agents, database, n_episodes=10, train=True,
            max_episode_steps=2000, train_each=1, update_database=True, 
            render=False, store_video=False, wandb_project=False, 
            save_model=True, save_model_each=50, MODEL_PATH='', 
            save_step_each=2, greedy_sampling=False, initialization=True,
            init_buffer_size=500, n_step_td=2, eval_each=5, train_n_MC=2,
            rest_n_MC=1, eval_MC=False, dual_eval=True, reward_scale_MC=1.0, 
            MC_version=1):

        best_return = -np.infty
        MC_period = train_n_MC + rest_n_MC
        MC_counter = 0

        if store_video:
            video = cv2.VideoWriter(video_folder+env.spec.id+'.mp4', 0x7634706d, 20, (1024, 1024))

        initialized = not (initialization and train)
        returns = []
        for episode in range(0, n_episodes):
            state_buffer = collections.deque(maxlen=n_step_td)
            action_buffer = collections.deque(maxlen=n_step_td)
            reward_buffer = collections.deque(maxlen=n_step_td)

            trajectory_buffer = ExperienceBuffer(max_episode_steps, level=4)
            train_MC_episode = train and (MC_counter < train_n_MC)

            step_counter = 0
            episode_done = False
            state = env.reset()
            episode_return = 0.0

            state_buffer.append(state)

            while not episode_done:
                if initialized:
                    if train_MC_episode or (not train and eval_MC):
                        skill, dist = agents[-1].sample_action_from_concept(state, explore=(not greedy_sampling))
                    else:
                        skill, dist = agents[-1].sample_action(state, explore=(not greedy_sampling))
                else:
                    skill = np.random.randint(agents[-1]._n_actions)
                    dist = np.ones(agents[-1]._n_actions) / agents[-1]._n_actions
                
                if render:
                    env.render()
                next_state, reward, done, info = self.second_level_step(env, agents[-1], state, skill)

                action_buffer.append(skill)
                reward_buffer.append(reward)
                dist = (dist + 1e-6) / (dist+1e-6).sum()
                entropy = -(dist * np.log(dist)).sum()
                entropy_baseline = self.optimizer.H_mean
                if entropy_baseline is None:
                    entropy_baseline = entropy
                entropy_difference = entropy - entropy_baseline
                alpha = agents[-1].second_level_architecture.get_alpha()
                gamma_n = gamma = self.optimizer.discount_factor
                for previous_step in range(0, len(reward_buffer)-1):
                    reward_buffer[-2-previous_step] += gamma_n * (reward + alpha * entropy_difference)
                    gamma_n *= gamma

                if store_video:
                    img_1 = env.sim.render(width=1024, height=512, depth=False, camera_name='front_camera')[::-1,:,:]
                    img_2 = env.sim.render(width=512, height=512, depth=False, camera_name='global_camera')[::-1,:,:]
                    img_3 = env.sim.render(width=512, height=512, depth=False, camera_name='global_camera_2')[::-1,:,:]
                    #assert img_1.shape == img_2.shape, 'Incompatible dimensions: img1:' + str(img_1.shape) + ', img2:' + str(img_2.shape)
                    img_up = np.concatenate((img_2, img_3), axis=1)
                    img = np.concatenate((img_up, img_1), axis=0)
                    video.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                buffer_ready = len(state_buffer) == n_step_td
                if buffer_ready:
                    initial_state = state_buffer[0]
                    initial_skill = action_buffer[0]
                    n_step_reward = reward_buffer[0]                    
                    
                    should_save_step = update_database and ((step_counter % save_step_each) == 0)
                    if should_save_step:
                        step = PixelExperienceSecondLevel(initial_state, 
                                                    initial_skill, n_step_reward, done, next_state)
                        database.append(step)

                    step = PixelExperienceThirdLevel(
                        initial_state, initial_skill, dist, n_step_reward, done, next_state)

                    trajectory_buffer.append(step)

                episode_return += reward
                state = next_state.copy()
                state_buffer.append(state)

                step_counter += 1

                should_train_in_this_step = train and ((step_counter % train_each) == 0) and initialized 
                if should_train_in_this_step:
                    metrics = self.optimizer.optimize(agents, database, n_step_td)
                    if wandb_project and metrics is not None:
                        metrics['step'] = step_counter
                        wandb.log(metrics)

                # should_train_in_this_step = train and ((step_counter % train_each) == 0) and initialized 
                # if should_train_in_this_step:
                #     metrics_l2 = self.optimizer.optimize(agents, database, n_step_td)
                #     metrics_l3 = self.optimizer.optimize_tabular(agents[-1], database)
                #     if metrics_l2 is not None and metrics_l3 is not None:
                #         metrics = {**metrics_l2, **metrics_l3}
                #     elif metrics_l2 is not None and metrics_l3 is None:
                #         metrics = metrics_l2
                #     else:
                #         metrics = metrics_l3
                #     if wandb_project and metrics is not None:
                #         metrics['step'] = step_counter
                #         wandb.log(metrics)
                
                if step_counter >= max_episode_steps or done:
                    episode_done = True
                
                initialized = initialized or (database.__len__() > init_buffer_size)

            returns.append(episode_return)

            if wandb_project and train:
                wandb.log({'episode': episode, 'return': episode_return})

            if save_model and ((episode + 1) % save_model_each == 0):
                agents[-1].save(MODEL_PATH + env.spec.id + '/')
            
            if train and (episode_return > best_return):
                best_return = episode_return
                agents[-1].save(MODEL_PATH + env.spec.id + '/', best=True)
            
            if train and ((episode+1) % eval_each == 0):
                eval_returns = self.loop(env, agents, None, n_episodes=1, train=False, 
                    max_episode_steps=max_episode_steps, update_database=False,
                    render=False, store_video=False, wandb_project=wandb_project,
                    save_model=False, greedy_sampling=True, initialization=False,
                    n_step_td=1, eval_MC=(eval_MC and not dual_eval))
                eval_metrics = {'episode_eval': episode//eval_each, 'eval_return': eval_returns.mean(), 'eval_return_std': eval_returns.std()}
                if dual_eval:
                    eval_returns_MC = self.loop(env, agents, None, n_episodes=1, train=False, 
                        max_episode_steps=max_episode_steps, update_database=False,
                        render=False, store_video=False, wandb_project=wandb_project,
                        save_model=False, greedy_sampling=True, initialization=False,
                        n_step_td=1, eval_MC=True)
                    eval_metrics['MC_eval_return'] = eval_returns_MC.mean()
                    eval_metrics['MC_eval_return_std'] = eval_returns_MC.std()
                wandb.log(eval_metrics)
            
            if train_MC_episode:
                last_MC_episode = (MC_counter + 1) == train_n_MC
                high_level_metrics = self.optimizer.optimize_tabular(agents[-1], trajectory_buffer, last_MC_episode, reward_scale_MC, version=MC_version)
                if wandb_project and high_level_metrics is not None:
                    wandb.log(high_level_metrics)
            
            MC_counter = (MC_counter+1) % MC_period

        return_array = np.array(returns)

        if store_video:
            video.release()

        if render:
            env.close()

        return return_array    

    def second_level_step(self, env, agent, state, skill):
        n_steps = agent._temporal_ratio
        first_level_step_counter = 0
        loop_reward = 0.0
        loop_done = False
        finished_loop = False

        while not finished_loop:
            action = agent.sample_first_level_action(state, skill)
            next_state, reward, done, info = env.step(action)
            loop_reward += reward
            loop_done = loop_done or done
            first_level_step_counter += 1            
            finished_loop = loop_done or ((first_level_step_counter % n_steps) == 0)
            state = next_state.copy()  

        return next_state, loop_reward, loop_done, info