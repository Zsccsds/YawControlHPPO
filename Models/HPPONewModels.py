from RegularWindPPO.backup.PPOFCNModel import PPO

import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical

class ACFCN(nn.Module):
    """
    Actor-Critic Feature Convolutional Network - a shared backbone network that generates features
    used by both the actor and critic networks
    """
    def __init__(self, state_dim, out_dim,hidden_layer, n_latent_var):
        """
        Initialize the ACFCN network

        Args:
            state_dim: Dimension of input state
            out_dim: Output dimension (actions or value)
            hidden_layer: Number of hidden layers
            n_latent_var: Size of latent variable/dimension of hidden layers
        """
        super(ACFCN, self).__init__()
        if hidden_layer==1:
            self.backbone = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
            )
        if hidden_layer==2:
            self.backbone = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
            )
        # Action output network - outputs probability distribution over actions
        self.fcn_action = nn.Sequential(
            nn.Linear(n_latent_var, out_dim),
            nn.Softmax(dim=-1),)
        # Period output network - outputs probability distribution over periods (15 options)
        self.fcn_peroid = nn.Sequential(
            nn.Linear(n_latent_var, 15),
            nn.Softmax(dim=-1),)

    def forward(self,state):
        """
       Forward pass through the network

       Args:
           state: Input state tensor

       Returns:
           action_probs: Probability distribution over actions
           period_probs: Probability distribution over periods
        """
        feature = self.backbone(state)
        return self.fcn_action(feature),self.fcn_peroid(feature)

class ActorCritic(nn.Module):
    """
    Actor-Critic neural network combining policy network (actor) and value network (critic)
    """
    def __init__(self, state_dim, action_dim,hidden_layer, hidden_size,device):
        """
        Initialize the Actor-Critic network

        Args:
            state_dim: Dimension of input state
            action_dim: Number of possible actions
            hidden_layer: Number of hidden layers in backbone
            hidden_size: Size of hidden layers
            device: Device to run the network on (CPU/GPU)
        """
        super(ActorCritic, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.action_net = ACFCN(state_dim, action_dim,hidden_layer, hidden_size).to(device)
        self.value_net = ACFCN(state_dim, 1, hidden_layer, hidden_size).to(device)

    def forward(self):
        raise NotImplementedError

    def train_act(self,state,memory):
        """
        Perform action selection during training (with sampling from probability distribution)

        Args:
            state: Current state
            memory: Memory buffer to store experience

        Returns:
            action: Selected action
            period: Selected period
        """
        state = torch.from_numpy(state).float().to(self.device)
        action_probs,period_probs = self.action_net(state)

        # Sample action from action probability distribution
        dist = Categorical(action_probs)
        action = dist.sample()
        # Sample period from period probability distribution
        dist_period = Categorical(period_probs)
        period = dist_period.sample()

        # Store experience in memory
        memory.states.append(state)
        memory.actions.append(action)
        memory.periods.append(period)
        memory.logprobs.append(dist.log_prob(action) + dist_period.log_prob(period))

        # Convert tensors back to numpy arrays
        action = action.detach().cpu().numpy()
        period = period.detach().cpu().numpy()
        return action,period

    def eval_act(self,state):
        """
        Perform action selection during evaluation (deterministic - take most likely action)

        Args:
            state: Current state

        Returns:
            action: Deterministic selected action (highest probability)
            period: Deterministic selected period (highest probability)
        """
        state = torch.from_numpy(state).float().to(self.device)
        action_probs,period_probs = self.action_net(state)
        # Take action with highest probability (deterministic)
        action = torch.max(action_probs, 1)[1]
        action = action.detach().cpu().numpy()
        # Take period with highest probability (deterministic)
        period = torch.max(period_probs, 1)[1]
        period = period.detach().cpu().numpy()
        return action,period

    def evaluate(self, state, action, peroid):
        """
        Evaluate state-action pairs for policy gradient computation

        Args:
           state: State tensor
           action: Action tensor
           peroid: Period tensor

        Returns:
           logprobs: Log probabilities of taken actions
           state_value: Estimated state value
           dist_entropy: Entropy of action distribution (for exploration bonus)
        """
        action_probs,period_probs = self.action_net(state)
        dist = Categorical(action_probs)
        dist_period = Categorical(period_probs)

        # Calculate log probabilities of the actions that were take
        logprobs = dist.log_prob(action) + dist_period.log_prob(peroid)
        dist_entropy = dist.entropy() + dist_period.entropy()

        # Get state value estimate
        state_value,_ = self.value_net(state)
        state_value = torch.squeeze(state_value)
        return logprobs, state_value, dist_entropy

class PPO2:
    """
    Proximal Policy Optimization algorithm implementation
    """
    def __init__(self, policy, policy_old, lr, betas, gamma, K_epochs, eps_clip, device):
        """
        Initialize PPO agent

        Args:
            policy: Current policy network
            policy_old: Old policy network (for importance sampling)
            lr: Learning rate
            betas: Beta parameters for Adam optimizer
            gamma: Discount factor
            K_epochs: Number of epochs to update the policy
            eps_clip: Clipping parameter for PPO
            device: Device to run computations on
        """
        self.lr = lr
        self.betas = betas  # betas
        self.gamma = gamma  # gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        # Initialize policy networks
        self.policy = policy
        self.policy_old = policy_old
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Optimizer for updating policy parameters
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)  # 优化器
        self.MseLoss = nn.MSELoss()

    def setlr(self,lr,betas):
        """
        Set new learning rate and beta parameters for optimizer

        Args:
            lr: New learning rate
            betas: New beta parameters
        """
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)  # 优化器

    def update(self, memory):
        """
        Update the policy using collected experiences in memory

        Args:
            memory: Experience memory containing states, actions, rewards, etc.
        """
        # Compute discounted rewards
        rewards = []
        discounted_reward = np.zeros(len(memory.is_terminals[0]))
        # Process rewards in reverse order to compute discounted returns
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            discounted_reward[is_terminal] = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Convert rewards to tensor and normalize
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device).view(-1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # Prepare batch data
        old_states = torch.stack(memory.states).to(self.device).detach()

        # Reshape states based on their dimensions
        if len(old_states.shape) == 3:
            old_states = old_states.view(-1, old_states.shape[2])
        if len(old_states.shape) == 4:
            old_states = old_states.view(-1,old_states.shape[2],old_states.shape[3])
        old_actions = torch.stack(memory.actions).to(self.device).detach().view(-1)
        old_periods = torch.stack(memory.periods).to(self.device).detach().view(-1)
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach().view(-1)

        # Perform multiple updates (K epochs)
        for _ in range(self.K_epochs):
            # Evaluate old actions and values at current policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_periods)

            # Calculate ratio of new and old policy probabilities
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Calculate advantage
            advantages = rewards - state_values.detach()

            # Calculate surrogate objectives for PPO
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # Total loss: policy loss + value loss + entropy bonus
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # Perform backpropagation
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        # Print average loss
        print('loss:{:.2f}'.format(loss.mean().item()))
        # Update old policy to match current policy
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    """
    Experience replay memory to store agent's experiences during training
    """
    def __init__(self):
        # Initialize empty lists to store different types of experiences

        self.actions = []      # Actions taken
        self.periods = []      # Periods selected
        self.states = []       # States observed
        self.logprobs = []     # Log probabilities of actions
        self.rewards = []      # Rewards received
        self.is_terminals = [] # Terminal state indicators

    def clear_memory(self):
        """
        Clear all stored experiences from memory
        """
        del self.actions[:]
        del self.periods[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def build_model(config,device):
    """
    Build and initialize the PPO model with specified configuration

    Args:
        config: Configuration dictionary containing hyperparameters
        device: Device to run the model on

    Returns:
        ppo: PPO agent
        memory: Experience memory object
    """
    # Extract hyperparameters from config
    action_dim = config['Net']['ActionDim']
    hidden_layer = config['Net']['HiddenLayer']
    hidden_size  = config['Net']['HiddenSize']

    # PPO-specific hyperparameters
    lr = config['PPO']['LR']
    betas = config['PPO']['BETAS']
    gamma = config['PPO']['GAMMA']
    K_epochs = config['PPO']['K_EPOCHS']
    eps_clip = config['PPO']['EPS_CLIP']

    # Create policy network (state dimension is hardcoded as 8)
    policy = ActorCritic(8,action_dim,hidden_layer, hidden_size, device)

    # Load pre-trained weights if specified
    if not config['Net']['Pretrain']== 'none':
        policy.load_state_dict(torch.load(config['Net']['Pretrain']))

    # Create old policy network (for PPO algorithm)
    policy_old = ActorCritic(8,action_dim, hidden_layer, hidden_size, device)

    # Create PPO agent
    ppo = PPO2(policy, policy_old,lr, (betas[0],betas[1]), gamma, K_epochs, eps_clip, device)
    memory = Memory()# Create experience memory
    return ppo,memory



if __name__=='__main__':
    net =1
    ppo = PPO()