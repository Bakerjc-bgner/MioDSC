from argparse import Action
import random
from typing_extensions import Self
from network import Classifier
import numpy as np
from copy import deepcopy
import torch
from network import OptionPolicyNetwork
class Option(object):
    def __init__(self, args, name, goal_state,sub_goal_reward=0, parent=None, timeout=1, seed=0,
                 num_subgoal_hits_required=3):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.seed = seed
        self.name = name
        self.parent = parent
        self.sub_goal_reward = sub_goal_reward
        self.timeout = timeout
        self.Term_set = []
        self.Init_set = []
        # 初始集合分类器
        self.classifier = Classifier(input_size=1, hidden_size=32)
        self.internal_policy = OptionPolicyNetwork(self.args)
        
    def is_termination_state(self,state):
    # Check if the state is a termination state
        if state in self.args.termination_states:
            return True
        else:
            return False

    def is_initiation_state(self,state_trajectory):
    # Check if the state is a termination state
       # Convert state_trajectory to a PyTorch tensor
        x = torch.tensor(state_trajectory, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        
        # Use the classifier to predict whether the first state is in the initiation set
        with torch.no_grad():
            output = self.classifier(x)
            is_in_initiation_set = output >= 0.5
        
        return is_in_initiation_set.item()

    def is_goal_state(self,state):
        # Check if the state matches the goal state
        if state == self.args.goal_state:
            return True
        else:
            return False
    
    def classifier_is_trained(self):
        if self.get_training_phase()== 'trained':
            return True
        return False

    def choose_action_via_option(self,obs,agent_num, avail_actions, state):
        """Option 的内部策略，输入状态和观察输出一个动作"""
        avail_actions_ind = np.nonzero(avail_actions)[0]
        action = np.random.choice(avail_actions_ind)
        
        if self.is_initiated(state):
            option_state = state
            terminated = False
            
            while not terminated:
                action_probs = self.internal_policy(option_state)
                action = torch.multinomial(action_probs, num_samples=1).item()
                
                # Update the state and termination condition
                option_state, terminated = self.update(option_state, action)
        
        return action
        

    