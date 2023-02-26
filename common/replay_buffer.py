import numpy as np
import threading

class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit
        self.n_options = 1
        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
                        's': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'r': np.empty([self.size, self.episode_limit, 1]),
                        'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        's_next': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'avail_u_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'padded': np.empty([self.size, self.episode_limit, 1]),
                        'terminated': np.empty([self.size, self.episode_limit, 1]),
                        'option':np.empty([self.size, self.episode_limit,self.n_agents,1])
                        }
        if self.args.alg == 'maven':
            self.buffers['z'] = np.empty([self.size, self.args.noise_dim])
        # thread lock
        self.lock = threading.Lock()

        # store the episode
    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]  # episode_number
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_next'][idxs] = episode_batch['o_next']
            self.buffers['s_next'][idxs] = episode_batch['s_next']
            self.buffers['avail_u'][idxs] = episode_batch['avail_u']
            self.buffers['avail_u_next'][idxs] = episode_batch['avail_u_next']
            self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']
            if self.args.alg == 'maven':
                self.buffers['z'][idxs] = episode_batch['z']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
    
    def estimate_state_trajectory_distribution(replay_buffer, batch_size):
        """
        Estimates the state trajectory distribution from a replay buffer.

        Parameters:
        -----------
        replay_buffer : ReplayBuffer object
            The replay buffer containing the state-action pairs encountered during training.
        num_samples : int, optional
            The number of state samples to draw from the replay buffer. Default is 100000.

        Returns:
        --------
        state_trajectory_distribution : numpy array
            The estimated state trajectory distribution represented as a probability density function (PDF).
            The array has shape (num_states,), where num_states is the total number of states in the environment.
        """
        # Sample state-action pairs from the replay buffer
        states, _, _, _, _ = replay_buffer.sample(batch_size)

        # Count the frequency of each state in the samples
        state_counts = np.zeros(replay_buffer.observation_space.n)
        for state in states:
            state_counts[state] += 1

        # Normalize the state counts to obtain the state trajectory distribution as a PDF
        state_trajectory_distribution = state_counts / (batch_size * replay_buffer.observation_space.n)

        return state_trajectory_distribution
    
    
    def compute_option_sample_distribution(self, data, num_options):
        option_counts = self.buffers['option'].shape[0]
        total_count = 0
        
        for episode in data:
            for step in episode:
                option_counts[step['option']] += 1
                total_count += 1
        
        option_probs = []
        for i in range(num_options):
            option_probs.append(option_counts[i] / total_count)
        
        return option_probs
    
    
    def mutual_information(self,p_s, p_w):
        """
        Computes the mutual information of two random variables.

        Parameters:
        -----------
        p_s : numpy array
            The probability distribution of the first random variable.
        p_w : numpy array
            The probability distribution of the second random variable.

        Returns:
        --------
        mutual_info : float
            The mutual information of the two random variables.
        """

        # Compute the joint probability distribution
        joint_prob = np.outer(p_s, p_w)

        # Compute the marginal probability distributions
        marg_prob1 = np.sum(joint_prob, axis=1)
        marg_prob2 = np.sum(joint_prob, axis=0)

        # Compute the entropy of each variable
        entropy1 = -np.sum(marg_prob1 * np.log2(marg_prob1 + 1e-12))
        entropy2 = -np.sum(marg_prob2 * np.log2(marg_prob2 + 1e-12))

        # Compute the joint entropy of the two variables
        joint_entropy = -np.sum(joint_prob * np.log2(joint_prob + 1e-12))

        # Compute the mutual information
        mutual_info = entropy1 + entropy2 - joint_entropy

        return mutual_info
   
        