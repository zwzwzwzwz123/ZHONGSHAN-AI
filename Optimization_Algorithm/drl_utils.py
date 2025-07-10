from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import torch.nn.functional as F

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
    

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, epsilon=0.01, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta=beta
        self.tree = SumTree(capacity)
        self.data = []
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        priority = (self.max_priority + self.epsilon) ** self.alpha
        self.tree.add(priority, data)

    def sample(self, batch_size):
        batch = []
        segment = self.tree.total() / batch_size
        priorities = []
        idxs = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return idxs, batch, is_weight, np.array(priorities)

    def update_priorities(self, indices, priorities):
        for i, priority in zip(indices, priorities):
            self.tree.update(i, (priority + self.epsilon) ** self.alpha)

    def size(self):
        return self.tree.n_entries

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class BDQnet(torch.nn.Module):
    """
    # Example usage:
    state_dim = 4
    hidden_dims = [64, 64, 32]
    action_dims = [2, 3, 4, 2, 3, 4, 2, 3, 4, 2]

    model = BranchingDQnet(state_dim, hidden_dims, action_dims)
    """
    def __init__(self, state_dim, hidden_dims, action_dims):
        super(BDQnet, self).__init__()
        self.hidden_layers = torch.nn.ModuleList()
        
        # 创建共享部分的隐藏层
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(torch.nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        # 创建分离部分的隐藏层和输出层
        self.branches = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dims[-1], hidden_dims[-1]),  # 分离隐藏层
                torch.nn.Linear(hidden_dims[-1], action_dim)        # 分离输出层
            ) for action_dim in action_dims
        ])
        
        # 创建价值函数的隐藏层和输出层
        self.value_branch = torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            torch.nn.Linear(hidden_dims[-1], 1)
        )

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        value = self.value_branch(x)
        advantages = [branch(x) for branch in self.branches]
        
        q_values = [value + advantage - advantage.mean(1, keepdim=True) for advantage in advantages]
        
        return q_values
    
class BDQ_Agent:
    def __init__(self,
                 state_dim,
                 hidden_dims, action_dims,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update,
                 device,
                 dqn_type='DuelingDQN'):# Default or 'DuelingDQN':BDQ
                                        # 'DoubleDQN':BD3QN
        
        self.action_dims = action_dims
        self.action_num = len(action_dims)
        self.q_net = BDQnet(state_dim, hidden_dims, action_dims).to(device)
        self.target_q_net = BDQnet(state_dim, hidden_dims, action_dims).to(device)
    
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device

    def take_action(self, state):
        action_branch_list = [0 for _ in range(self.action_num)]
        if np.random.random() < self.epsilon:
            for i in range(self.action_num):
                action_branch_list[i] = np.random.randint(self.action_dims[i]) #跟据该分支动作数决定随机数范围
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            q_values_list = self.q_net(state)
            for index, element in enumerate(q_values_list):
                action_branch_list[index] = element.argmax().item()
        return action_branch_list
    
    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        q_value_branch_list = self.q_net(state)
        max_q_value_list = [q_value_branch.max().item() for q_value_branch in q_value_branch_list]
        return max_q_value_list

    def update(self, transition_dict, is_weight):
        is_weight = torch.tensor(is_weight).to(self.device)
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        actions_list = [actions[:, i] for i in range(self.action_num)]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        q_values_list = self.q_net(states)
        if self.dqn_type == 'DoubleDQN':
            target_q_values_list = self.target_q_net(next_states)
            best_actions_list = [q.argmax(dim=1, keepdim=True) for q in q_values_list]
            target_q_values_list = [target_q.gather(1, best_actions_list[i]) for i, target_q in enumerate(target_q_values_list)]
        else:
            max_next_q_values_list = self.target_q_net(next_states)
        
        q_targets_list = []
        dqn_losses = []

        for i in range(self.action_num):
            q_values_list[i] = q_values_list[i].gather(1, actions_list[i].reshape(states.shape[0], 1).long())
            if self.dqn_type == 'DoubleDQN':
                q_targets_tmp = rewards + self.gamma * target_q_values_list[i] * (1 - dones)
            else:
                max_next_q_values_list[i] = max_next_q_values_list[i].max(1)[0].view(-1, 1)
                q_targets_tmp = rewards + self.gamma * max_next_q_values_list[i] * (1 - dones)
            q_targets_list.append(q_targets_tmp)
            dqn_loss_tmp = torch.mean(is_weight * F.mse_loss(q_values_list[i], q_targets_list[i]))
            dqn_losses.append(dqn_loss_tmp)
        
        dqn_loss = torch.stack(dqn_losses).sum()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
        
    def compute_td_error(self, transition_dict, is_weight):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        actions_list = [actions[:, i] for i in range(self.action_num)]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values_list = self.q_net(states)
        q_targets_list = []
        
        for i in range(self.action_num):
            q_values_list[i] = q_values_list[i].gather(1, actions_list[i].reshape(states.shape[0], 1).long())
        
        with torch.no_grad():
            max_next_q_values_list = self.target_q_net(next_states)
            for i in range(self.action_num):
                max_next_q_values_list[i] = max_next_q_values_list[i].max(1)[0].view(-1, 1)
                q_targets_tmp = rewards + self.gamma * max_next_q_values_list[i] * (1 - dones)
                q_targets_list.append(q_targets_tmp)
        
        td_errors_list = []
        for i in range(self.action_num):
            td_errors_list.append((q_values_list[i] - q_targets_list[i]).abs())
        
        weighted_td_errors = (torch.tensor(is_weight).reshape(states.shape[0], 1).to(self.device) * sum(td_errors_list)).squeeze().to('cpu').detach().numpy()

        return weighted_td_errors

# 保存模型函数
def save_model(agent, filepath):
    torch.save(agent.q_net.state_dict(), filepath)
    torch.save(agent.target_q_net.state_dict(), filepath + "_target")

# 加载模型函数
def load_model(agent, filepath, device):
    agent.q_net.load_state_dict(torch.load(filepath, map_location=device))
    agent.target_q_net.load_state_dict(torch.load(filepath + "_target", map_location=device))
    agent.q_net.to(device)
    agent.target_q_net.to(device)

def train_BDQ(agent, env, num_episodes:int, replay_buffer, minimal_size:int, batch_size:int, save_path:str, save_interval:int):
    '''train的结果可以直接作为output'''
    return_list = []
    max_q_value_lists_list=[[] for i in range(agent.action_num)]
    max_reward = -10000

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    # 选个每个分支的动作
                    action_branch_list = agent.take_action(state)
                    max_q_value_list = agent.max_q_value(state)
                    for i in range(agent.action_num):
                        max_q_value_lists_list[i].append(max_q_value_list[i])
                        
                    next_state, reward, done, _ = env.step(action_branch_list)
                    replay_buffer.add(state,action_branch_list,reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    
                    #记录最佳reward
                    if reward>max_reward:
                        max_reward = reward
                        best_action = action_branch_list

                    if replay_buffer.size() > minimal_size:
                        # 从优先级经验回放缓冲区中采样
                        idxs, b, is_weight, p =replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': [item[0] for item in b],
                            'actions': [item[1] for item in b],
                            'next_states': [item[3] for item in b],
                            'rewards': [item[2] for item in b],
                            'dones': [item[4] for item in b]
                        }
                    
                        td_errors = np.abs(agent.compute_td_error(transition_dict, is_weight))
                        replay_buffer.update_priorities(idxs, td_errors)

                        agent.update(transition_dict, is_weight)

                return_list.append(episode_return)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })

                pbar.update(1)

                # 定期保存模型
                if save_path:
                    if (i_episode + 1) % save_interval == 0:
                        save_model(agent, save_path)
    if save_path:
        print(f"模型已保存至 {save_path}")
    # return return_list, max_q_value_lists_list, max_reward, best_action # 只有best_action是重要的
    return best_action

def infer_action(agent, state):
    state = torch.tensor([state], dtype=torch.float).to(agent.device)
    q_values_list = agent.q_net(state)
    action_branch_list = [q.argmax().item() for q in q_values_list]
    return action_branch_list