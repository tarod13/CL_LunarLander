import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from policy_optimizers import Optimizer
from utils import one_hot_embedding


class S_ConceptOptimizer(Optimizer):
    def __init__(self, batch_size=64, beta=0.0, eta=0.0, n_batches_estimation=2,
        update_rate=0.05, consider_task=True, detach_logs=True, clip_value=1.0):
        super().__init__()
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.beta = beta
        self.eta = eta
        self.n_batches_estimation = n_batches_estimation
        self.update_rate = update_rate
        self.consider_task = consider_task
        self.detach_logs = detach_logs
        self.PAST = None
        self.PAS = None
        self.PAST_test = None
        self.PAS_test = None
        
    def optimize(self, agent, database, n_actions, n_tasks, initialization=False, train=True): 
        if database.__len__() < self.batch_size:
            return None

        if train:
            PAST = self.PAST
            PAS = self.PAS
        else:
            PAST = self.PAST_test
            PAS = self.PAS_test

        if (self.n_batches_estimation > 1) or initialization:
            if not initialization:
                n_batches = self.n_batches_estimation
            else:
                n_batches = 200
                # print("Initializing estimation")
            # Estimate visits
            NAST = None

            with torch.no_grad():
                for batch in range(0, n_batches):
                    # Sample batch        
                    states, actions, rewards, dones, \
                        next_states, tasks = \
                        database.sample(self.batch_size)
                    
                    PS_s, log_PS_s = agent(states)
                    A_one_hot = one_hot_embedding(actions, n_actions)
                    T_one_hot = one_hot_embedding(tasks, n_tasks)

                    NAS_batch = PS_s.unsqueeze(2) * A_one_hot.unsqueeze(1)
                    NAST_batch = torch.einsum('ijk,ih->hjk', NAS_batch, T_one_hot).detach()

                    if NAST is None:
                        NAST = NAST_batch
                    else:
                        NAST = NAST + NAST_batch
            
            NAST += 1e-8

        # Sample batch        
        states, actions, rewards, dones, \
            next_states, tasks = \
            database.sample(self.batch_size)
        
        PS_s, log_PS_s = agent(states)
        A_one_hot = one_hot_embedding(actions, n_actions)
        T_one_hot = one_hot_embedding(tasks, n_tasks)
        
        if (self.n_batches_estimation == 1) and not initialization:
            if self.detach_logs:
                PAS_batch = PS_s.detach().unsqueeze(2) * A_one_hot.unsqueeze(1)
            else:
                PAS_batch = PS_s.unsqueeze(2) * A_one_hot.unsqueeze(1)
            NAST = torch.einsum('ijk,ih->hjk', PAS_batch, T_one_hot) + 1e-20
            NAS = PAS_batch.sum(0) + 1e-20
        else:
            NAS = NAST.sum(0)
        
        PAST_batch = NAST / NAST.sum()
        PAS_batch = NAS / NAS.sum()

        if PAST is None:
            PAST = PAST_batch
        else:
            PAST = PAST * (1.-self.update_rate) + PAST_batch * self.update_rate

        if PAS is None:
            PAS = PAS_batch
        else:
            PAS = PAS * (1.-self.update_rate) + PAS_batch * self.update_rate

        # PAST = NAST / NAST.sum()
        PT = PAST.sum(2).sum(1)
        PT = PT / PT.sum()
        PST = PAST.sum(2)
        PS_T = PST / PST.sum(1, keepdim=True)
        PA_ST = PAST / PAST.sum(2, keepdim=True)
        PAT = PAST.sum(1)
        PA_T = PAT / PAT.sum(1, keepdim=True)
        PAS_T = PAST / PAST.sum(2, keepdim=True).sum(1, keepdim=True)
        PA = PAS.sum(0)
        PS = PAS.sum(1)
        PA_S = PAS / PAS.sum(1, keepdim=True)

        log_PS_T = torch.log(PS_T)
        log_PA_T = torch.log(PA_T)
        log_PA_ST = torch.log(PA_ST) 
        log_PA = torch.log(PA)
        log_PS = torch.log(PS)
        log_PA_S = torch.log(PA_S)                
        
        T_one_hot_dist = (T_one_hot + 1e-20) / (T_one_hot + 1e-20).sum(0, keepdim=True)
        PS_sgT = PS_s.detach().unsqueeze(1) * T_one_hot_dist.unsqueeze(2)
        HS_gT_samp = torch.einsum('ihj,hj->ih', PS_sgT, -log_PS_T).sum(0)
        HS_gT = torch.einsum('ij,ij->i', PS_s, -log_PS_T[tasks,:]).mean()
        HS_s = -(PS_s * log_PS_s).sum(1).mean()
        ISs_gT = HS_gT_samp - HS_s
        ISs_T = HS_gT - HS_s
        
        HA_gT = -(PA_T * log_PA_T).sum(1)
        HA_T = (PT * HA_gT).sum()
        HA_sT = 0.0*np.log(n_actions)
        assert log_PA_ST[tasks,:,actions].shape == PS_s.shape, 'Problems with dimensions (T)'
        HA_ST = -(PS_s * log_PA_ST[tasks,:,actions]).sum(1).mean()
        HA_SgT = -(PAS_T * log_PA_ST).sum((1,2))  

        HS = -(PS_s.mean(0) * log_PS).sum()
        ISs = HS - HS_s
        IST = HS - HS_gT

        HA_s = 0.0*np.log(n_actions)
        HA = -(PA * log_PA).sum()
        assert log_PA_S[:,actions].T.shape == PS_s.shape, 'Problems with dimensions (NT)'
        HA_S = -(PS_s * log_PA_S[:,actions].T).sum(1).mean()
        
        IAs_gT = HA_gT - HA_sT   
        IAS_gT = HA_gT - HA_SgT
        IAs_SgT = IAs_gT - IAS_gT

        IAs_T = (PT * IAs_gT).sum()
        IAS_T = HA_T - HA_ST
        IAs_ST = IAs_T - IAS_T         
        
        IAs = HA - HA_s
        IAS = HA - HA_S
        IAs_S = IAs - IAS

        n_concepts = PS_s.shape[1]
        H_max = np.log(n_concepts)
        if self.consider_task:
            classifier_loss = IAs_ST + self.beta * ISs_T + self.eta * IST
        else:
            classifier_loss = IAs_S + self.beta * ISs                

        if train:
            agent.classifier.optimizer.zero_grad()
            classifier_loss.backward()
            clip_grad_norm_(agent.classifier.parameters(), self.clip_value)
            agent.classifier.optimizer.step()

        if not self.detach_logs:
            PAST = PAST.detach()
            PAS = PAS.detach()
        
        if train:
            self.PAST = PAST
            self.PAS = PAS
        else:
            self.PAST_test = PAST 
            self.PAS_test = PAS 

        label = ''
        if not train:
            label += '_test'

        joint_metrics = {
            'HA'+label: HA.item(),
            'HA_S'+label: HA_S.item(),
            'HS'+label: HS.item(),
            'HS_T'+label: HS_gT.mean().item(),
            'HS_s'+label: HS_s.item(),
            'HA_T'+label: HA_T.item(),
            'HA_sT'+label: HA_sT,
            'HA_ST'+label: HA_ST.item(),
            'IST'+label: IST.item(),
            'ISs'+label: ISs.item(),
            'ISs_T'+label: ISs_T.item(),
            'IAs'+label: IAs.item(),
            'IAS'+label: IAS.item(),
            'IAs_S'+label: IAs_S.item(),
            'IAs_T'+label: IAs_T.item(),
            'IAS_T'+label: IAS_T.item(),
            'IAs_ST'+label: IAs_ST.item(),
            'loss'+label: classifier_loss.item(),
        }

        metrics_per_task = {}
        for task in range(0, n_tasks):
            metrics_per_task['HS_T'+str(task)+label] = HS_gT_samp[task].item()
            metrics_per_task['HA_T'+str(task)+label] = HA_gT[task].item()
            metrics_per_task['HA_ST'+str(task)+label] = HA_SgT[task].item()
            metrics_per_task['ISs_T'+str(task)+label] = ISs_gT[task].item()
            metrics_per_task['IAs_T'+str(task)+label] = IAs_gT[task].item()
            metrics_per_task['IAS_T'+str(task)+label] = IAS_gT[task].item()
            metrics_per_task['IAs_ST'+str(task)+label] = IAs_SgT[task].item()

        metrics = {**joint_metrics, **metrics_per_task}
        
        return metrics


class trajectory_ConceptOptimizer(Optimizer):
    def __init__(
        self, PA_ST:torch.tensor, batch_size=64, beta=0.0, eta=0.0, 
        n_batches_estimation=2, update_rate=0.05, clip_value=1.0
        ):
        super().__init__()
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.beta = beta
        self.eta = eta
        self.n_batches_estimation = n_batches_estimation
        self.update_rate = update_rate
        self.PST = None
        self.PST_test = None
        self.PA_ST = PA_ST.to(device)
        self.PA_ST.requires_grad = False
        
    def optimize(self, agent, database, n_actions, n_tasks, initialization=False, train=True): 
        if database.__len__() < self.batch_size:
            return None

        if train:
            PST = self.PST            
        else:
            PST = self.PST_test

        if (self.n_batches_estimation > 1) or initialization:
            if not initialization:
                n_batches = self.n_batches_estimation
            else:
                n_batches = 200
                print("Initializing estimation")
            # Estimate visits
            NST = None

            with torch.no_grad():
                for batch in range(0, n_batches):
                    # Sample batch        
                    states, actions, rewards, dones, \
                        next_states, tasks = \
                        database.sample(self.batch_size)
                    
                    PS_s, log_PS_s = agent(states)
                    T_one_hot = one_hot_embedding(tasks, n_tasks)

                    NST_batch = torch.einsum('ij,ih->hj', PS_s, T_one_hot).detach()

                    if NST is None:
                        NST = NST_batch
                    else:
                        NST = NST + NST_batch
            
            NST += 1e-8

        # Sample batch        
        states, actions, rewards, dones, \
            next_states, tasks = \
            database.sample(self.batch_size)
        
        PS_s, log_PS_s = agent(states)
        T_one_hot = one_hot_embedding(tasks, n_tasks)
        
        if (self.n_batches_estimation == 1) and not initialization:
            NST = torch.einsum('ij,ih->hj', PS_s, T_one_hot).detach() + 1e-20
        
        PST_batch = NST / NST.sum()
        
        if PST is None:
            PST = PST_batch
        else:
            PST = PST * (1.-self.update_rate) + PST_batch * self.update_rate

        PAST = PST.unsqueeze(2) * self.PA_ST
        PT = PST.sum(1)
        PT = PT / PT.sum()
        PS_T = PST / PST.sum(1, keepdim=True)
        PAT = PAST.sum(1)
        PA_T = PAT / PAT.sum(1, keepdim=True)
        PAS_T = PAST / PAST.sum(2).sum(1)
        PA = PAT.sum(0)
        PS = PST.sum(0)
        PAS = PAST.sum(0)
        PA_S = PAS / PAS.sum(1, keepdim=True)

        log_PS_T = torch.log(PS_T)
        log_PA_T = torch.log(PA_T)
        log_PA_ST = torch.log(self.PA_ST) 
        log_PA = torch.log(PA)
        log_PS = torch.log(PS)
        log_PA_S = torch.log(PA_S)                
        
        T_one_hot_dist = (T_one_hot + 1e-20) / (T_one_hot + 1e-20).sum(0, keepdim=True)
        PS_sgT = PS_s.detach().unsqueeze(1) * T_one_hot_dist.unsqueeze(2)
        HS_gT_samp = torch.einsum('ihj,hj->ih', PS_sgT, -log_PS_T).sum(0)
        HS_gT = torch.einsum('ij,ij->i', PS_s, -log_PS_T[tasks,:]).mean()
        HS_s = -(PS_s * log_PS_s).sum(1).mean()
        ISs_gT = HS_gT_samp - HS_s
        ISs_T = HS_gT - HS_s
        
        HA_gT = -(PA_T * log_PA_T).sum(1)
        HA_T = (PT * HA_gT).sum()
        HA_sT = 0.0*np.log(n_actions)
        assert log_PA_ST[tasks,:,actions].shape == PS_s.shape, 'Problems with dimensions (T)'
        HA_ST = -(PS_s * log_PA_ST[tasks,:,actions]).sum(1).mean()
        HA_SgT = -(PAS_T * log_PA_ST).sum((1,2))  

        HS = -(PS_s.mean(0) * log_PS).sum()
        ISs = HS - HS_s
        IST = HS - HS_gT

        HA_s = 0.0*np.log(n_actions)
        HA = -(PA * log_PA).sum()
        assert log_PA_S[:,actions].T.shape == PS_s.shape, 'Problems with dimensions (NT)'
        HA_S = -(PS_s * log_PA_S[:,actions].T).sum(1).mean()
        
        IAs_gT = HA_gT - HA_sT   
        IAS_gT = HA_gT - HA_SgT
        IAs_SgT = IAs_gT - IAS_gT

        IAs_T = (PT * IAs_gT).sum()
        IAS_T = HA_T - HA_ST
        IAs_ST = IAs_T - IAS_T         
        
        IAs = HA - HA_s
        IAS = HA - HA_S
        IAs_S = IAs - IAS

        n_concepts = PS_s.shape[1]
        H_max = np.log(n_concepts)
        
        classifier_loss = -torch.log((PS_s * self.PA_ST[tasks,:,actions]).sum(1) + 1e-10).mean()
        
        if train:
            agent.classifier.optimizer.zero_grad()
            classifier_loss.backward()
            clip_grad_norm_(agent.classifier.parameters(), self.clip_value)
            agent.classifier.optimizer.step()
       
        if train:
            self.PST = PST
        else:
            self.PST_test = PST 

        label = ''
        if not train:
            label += '_test'

        joint_metrics = {
            'HA'+label: HA.item(),
            'HA_S'+label: HA_S.item(),
            'HS'+label: HS.item(),
            'HS_T'+label: HS_gT.mean().item(),
            'HS_s'+label: HS_s.item(),
            'HA_T'+label: HA_T.item(),
            'HA_sT'+label: HA_sT,
            'HA_ST'+label: HA_ST.item(),
            'IST'+label: IST.item(),
            'ISs'+label: ISs.item(),
            'ISs_T'+label: ISs_T.item(),
            'IAs'+label: IAs.item(),
            'IAS'+label: IAS.item(),
            'IAs_S'+label: IAs_S.item(),
            'IAs_T'+label: IAs_T.item(),
            'IAS_T'+label: IAS_T.item(),
            'IAs_ST'+label: IAs_ST.item(),
            'loss'+label: classifier_loss.item(),
        }

        metrics_per_task = {}
        for task in range(0, n_tasks):
            metrics_per_task['HS_T'+str(task)+label] = HS_gT_samp[task].item()
            metrics_per_task['HA_T'+str(task)+label] = HA_gT[task].item()
            metrics_per_task['HA_ST'+str(task)+label] = HA_SgT[task].item()
            metrics_per_task['ISs_T'+str(task)+label] = ISs_gT[task].item()
            metrics_per_task['IAs_T'+str(task)+label] = IAs_gT[task].item()
            metrics_per_task['IAS_T'+str(task)+label] = IAS_gT[task].item()
            metrics_per_task['IAs_ST'+str(task)+label] = IAs_SgT[task].item()

        metrics = {**joint_metrics, **metrics_per_task}
        
        return metrics
        

class trajectory_ConceptOptimizer_v2(Optimizer):
    def __init__(self, PA_ST:torch.tensor, batch_size=64, beta=0.0, eta=0.0, n_batches_estimation=2,
        update_rate=0.05, consider_task=True, detach_logs=True, clip_value=1.0):
        super().__init__()
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.beta = beta
        self.eta = eta
        self.n_batches_estimation = n_batches_estimation
        self.update_rate = update_rate
        self.consider_task = consider_task
        self.detach_logs = detach_logs
        self.PAST = None
        self.PAS = None
        self.PAST_test = None
        self.PAS_test = None
        self.PA_ST = PA_ST.to(device)
        self.PA_ST.requires_grad = False
        
    def optimize(self, agent, database, n_actions, n_tasks, initialization=False, train=True): 
        if database.__len__() < self.batch_size:
            return None

        if train:
            PAST = self.PAST
            PAS = self.PAS
        else:
            PAST = self.PAST_test
            PAS = self.PAS_test

        if (self.n_batches_estimation > 1) or initialization:
            if not initialization:
                n_batches = self.n_batches_estimation
            else:
                n_batches = 200
                print("Initializing estimation (trajectory-v2)")
            # Estimate visits
            NAST = None

            with torch.no_grad():
                for batch in range(0, n_batches):
                    # Sample batch        
                    states, actions, rewards, dones, \
                        next_states, tasks = \
                        database.sample(self.batch_size)
                    
                    PS_s, log_PS_s = agent(states)
                    A_one_hot = one_hot_embedding(actions, n_actions)
                    T_one_hot = one_hot_embedding(tasks, n_tasks)

                    NAS_batch = PS_s.unsqueeze(2) * A_one_hot.unsqueeze(1)
                    NAST_batch = torch.einsum('ijk,ih->hjk', NAS_batch, T_one_hot).detach()

                    if NAST is None:
                        NAST = NAST_batch
                    else:
                        NAST = NAST + NAST_batch
            
            NAST += 1e-8

        # Sample batch        
        states, actions, rewards, dones, \
            next_states, tasks = \
            database.sample(self.batch_size)
        
        PS_s, log_PS_s = agent(states)
        A_one_hot = one_hot_embedding(actions, n_actions)
        T_one_hot = one_hot_embedding(tasks, n_tasks)
        
        if (self.n_batches_estimation == 1) and not initialization:
            if self.detach_logs:
                PAS_batch = PS_s.detach().unsqueeze(2) * A_one_hot.unsqueeze(1)
            else:
                PAS_batch = PS_s.unsqueeze(2) * A_one_hot.unsqueeze(1)
            NAST = torch.einsum('ijk,ih->hjk', PAS_batch, T_one_hot) + 1e-20
            NAS = PAS_batch.sum(0) + 1e-20
        else:
            NAS = NAST.sum(0)
        
        PAST_batch = NAST / NAST.sum()
        PAS_batch = NAS / NAS.sum()

        if PAST is None:
            PAST = PAST_batch
        else:
            PAST = PAST * (1.-self.update_rate) + PAST_batch * self.update_rate

        if PAS is None:
            PAS = PAS_batch
        else:
            PAS = PAS * (1.-self.update_rate) + PAS_batch * self.update_rate

        # PAST = NAST / NAST.sum()
        PT = PAST.sum(2).sum(1)
        PT = PT / PT.sum()
        PST = PAST.sum(2)
        PS_T = PST / PST.sum(1, keepdim=True)
        PA_ST = PAST / PAST.sum(2, keepdim=True)
        PAT = PAST.sum(1)
        PA_T = PAT / PAT.sum(1, keepdim=True)
        PAS_T = PAST / PAST.sum(2).sum(1)
        PA = PAS.sum(0)
        PS = PAS.sum(1)
        PA_S = PAS / PAS.sum(1, keepdim=True)

        log_PS_T = torch.log(PS_T)
        log_PA_T = torch.log(PA_T)
        log_PA_ST = torch.log(PA_ST) 
        log_PA = torch.log(PA)
        log_PS = torch.log(PS)
        log_PA_S = torch.log(PA_S)                
        
        T_one_hot_dist = (T_one_hot + 1e-20) / (T_one_hot + 1e-20).sum(0, keepdim=True)
        PS_sgT = PS_s.detach().unsqueeze(1) * T_one_hot_dist.unsqueeze(2)
        HS_gT_samp = torch.einsum('ihj,hj->ih', PS_sgT, -log_PS_T).sum(0)
        HS_gT = torch.einsum('ij,ij->i', PS_s, -log_PS_T[tasks,:]).mean()
        HS_s = -(PS_s * log_PS_s).sum(1).mean()
        ISs_gT = HS_gT_samp - HS_s
        ISs_T = HS_gT - HS_s
        
        HA_gT = -(PA_T * log_PA_T).sum(1)
        HA_T = (PT * HA_gT).sum()
        HA_sT = 0.0*np.log(n_actions)
        assert log_PA_ST[tasks,:,actions].shape == PS_s.shape, 'Problems with dimensions (T)'
        HA_ST = -(PS_s * log_PA_ST[tasks,:,actions]).sum(1).mean()
        HA_SgT = -(PAS_T * log_PA_ST).sum((1,2))  

        HS = -(PS_s.mean(0) * log_PS).sum()
        ISs = HS - HS_s
        IST = HS - HS_gT

        HA_s = 0.0*np.log(n_actions)
        HA = -(PA * log_PA).sum()
        assert log_PA_S[:,actions].T.shape == PS_s.shape, 'Problems with dimensions (NT)'
        HA_S = -(PS_s * log_PA_S[:,actions].T).sum(1).mean()
        
        IAs_gT = HA_gT - HA_sT   
        IAS_gT = HA_gT - HA_SgT
        IAs_SgT = IAs_gT - IAS_gT

        IAs_T = (PT * IAs_gT).sum()
        IAS_T = HA_T - HA_ST
        IAs_ST = IAs_T - IAS_T         
        
        IAs = HA - HA_s
        IAS = HA - HA_S
        IAs_S = IAs - IAS

        n_concepts = PS_s.shape[1]
        H_max = np.log(n_concepts)

        classifier_loss = -torch.log((PS_s * self.PA_ST[tasks,:,actions].detach()).sum(1)+1e-10).mean()                

        if train:
            agent.classifier.optimizer.zero_grad()
            classifier_loss.backward()
            clip_grad_norm_(agent.classifier.parameters(), self.clip_value)
            agent.classifier.optimizer.step()

        if not self.detach_logs:
            PAST = PAST.detach()
            PAS = PAS.detach()
        
        if train:
            self.PAST = PAST
            self.PAS = PAS
        else:
            self.PAST_test = PAST 
            self.PAS_test = PAS 

        label = ''
        if not train:
            label += '_test'

        joint_metrics = {
            'HA'+label: HA.item(),
            'HA_S'+label: HA_S.item(),
            'HS'+label: HS.item(),
            'HS_T'+label: HS_gT.mean().item(),
            'HS_s'+label: HS_s.item(),
            'HA_T'+label: HA_T.item(),
            'HA_sT'+label: HA_sT,
            'HA_ST'+label: HA_ST.item(),
            'IST'+label: IST.item(),
            'ISs'+label: ISs.item(),
            'ISs_T'+label: ISs_T.item(),
            'IAs'+label: IAs.item(),
            'IAS'+label: IAS.item(),
            'IAs_S'+label: IAs_S.item(),
            'IAs_T'+label: IAs_T.item(),
            'IAS_T'+label: IAS_T.item(),
            'IAs_ST'+label: IAs_ST.item(),
            'loss'+label: classifier_loss.item(),
        }

        metrics_per_task = {}
        for task in range(0, n_tasks):
            metrics_per_task['HS_T'+str(task)+label] = HS_gT_samp[task].item()
            metrics_per_task['HA_T'+str(task)+label] = HA_gT[task].item()
            metrics_per_task['HA_ST'+str(task)+label] = HA_SgT[task].item()
            metrics_per_task['ISs_T'+str(task)+label] = ISs_gT[task].item()
            metrics_per_task['IAs_T'+str(task)+label] = IAs_gT[task].item()
            metrics_per_task['IAS_T'+str(task)+label] = IAS_gT[task].item()
            metrics_per_task['IAs_ST'+str(task)+label] = IAs_SgT[task].item()

        metrics = {**joint_metrics, **metrics_per_task}
        
        return metrics
