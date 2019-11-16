import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras import layers
from keras import regularizers
from keras import models
from keras import optimizers
import keras.backend as K

import scipy.stats  # used in exp_buffer for PER.
import warnings  # for trainer for surpress keras warnings;

from datetime import datetime
from collections import defaultdict
import time
from tqdm import tqdm
import pickle


'''
DDPG_SER (Deep Disttibutional Deterministic Policy Gradient)
includes below additional for traditional DDPG
 - prioritized experience replay
 - stratified experience replay
 - n-step advantage TD calculation
 - Boltzmann explorer in Agent
 
 Stratified experience replay defines category of the experience based on info at the episode end.
 If 'info' from environment is None, this code does not work correctly.


'''

########################################################################################################################
class Environment(object):
    '''
    Dummy environment for testing.
    '''
    def __init__(self, n_agents=2, state_dims=[1, 3, 5], n_actions=5):
        self.n_agents = n_agents
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.reset()


    def step(self, action):
        for state in self.states:
            pass
        reward = np.random.randint(-1, 1+1, self.done.shape)
        self.done = np.random.binomial(1, 0.2, self.done.shape)

        return [self.state(), np.copy(reward), np.array(['X'] * n_agents).reshape(-1, 1), np.copy(self.done)]


    def state(self):
        return [np.copy(state) for state in self.states]


    def reset(self):
        self.states = []
        for state_dim in self.state_dims:
            self.states.append(np.arange(state_dim) * np.ones(shape=(self.n_agents, 1)))
        self.done = np.ones(shape=(self.n_agents, 1), dtype=bool)


########################################################################################################################
def gen_model_policy(state_shapes=[[10], [20]], n_actions=1, hidden_dims=[2, 4, 6],
                     reg_l1=0.0, reg_l2=0.0, batch_norm_input=False, batch_norm_hidden=False,
                     action_min=0.0, action_max=1.0,
                     state_reg=False, state_min=0, state_max=1):
    '''
    policy network
    :param state_shapes: tuple or list of state shape.
    :param n_actions: dimensions of actions.
    :param hidden_dims: list of hidden neurons in dense layers.
    :param reg_l1:
    :param reg_l2:
    :param action_min: minimum action value.
    :param action_max: maximum action value.
    :return:
    '''
    input_ts = [layers.Input(state_shape) for state_shape in state_shapes]
    if 2<=len(input_ts):
        input_concat = layers.concatenate(input_ts, axis=-1)
    else:
        input_concat = layers.Lambda(lambda x: x)(input_ts[0])

    if state_reg:
        input_concat = layers.Lambda(lambda x: (x - state_min) / (state_max - state_min))(input_concat)

    if 0<len(hidden_dims):
        for i, hidden_dim in enumerate(hidden_dims):
            if i==0:
                x = layers.Dense(hidden_dim,
                                 kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(input_concat)
                if batch_norm_input:
                    x = layers.normalization.BatchNormalization()(x)
                x = layers.LeakyReLU(alpha=0.1)(x)
            else:
                x = layers.Dense(hidden_dim,
                                 kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x)
                if batch_norm_hidden:
                    x = layers.normalization.BatchNormalization()(x)
                x = layers.LeakyReLU(alpha=0.1)(x)
    else:
        x = input_concat

    x = layers.Dense(n_actions, activation='tanh',
                     kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x)

    y = layers.Lambda(lambda x: action_min + (action_max-action_min) * (x+1)/2)(x)

    return models.Model(input_ts, y)


####################################################################################################################
def gen_model_value(state_shapes=[[10], [20]], n_actions=1,
                    hidden_action_dims=[10, 20], hidden_state_dims=[5, 6], hidden_dims=[32, 16],
                    reg_l1=0.0, reg_l2=0.0,
                    batch_norm_state_input=False, batch_norm_state_hidden=False,
                    batch_norm_action_input=False, batch_norm_action_hidden=False,
                    batch_norm_hidden=False,
                    state_reg=False, state_min=0, state_max=1,
                    action_reg=False, action_min=0, action_max=1,
                    value_reg=False, value_min=0, value_max=1):
    # input_state
    input_state = [layers.Input(state_shape) for state_shape in state_shapes]
    if 2<=len(input_state):
        input_state_concat = layers.concatenate(input_state, axis=-1)
    else:
        input_state_concat = layers.Lambda(lambda x: x)(input_state)

    if state_reg:
        input_state_concat = layers.Lambda(lambda x: (x - state_min) / (state_max - state_min))(input_state_concat)

    if 0 < len(hidden_state_dims):
        for i, hidden_dim in enumerate(hidden_state_dims):
            if i == 0:
                x = layers.Dense(hidden_dim,
                                 kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(input_state_concat)
                if batch_norm_state_input:
                    x = layers.normalization.BatchNormalization()(x)
                x = layers.LeakyReLU(alpha=0.1)(x)
            else:
                x = layers.Dense(hidden_dim,
                                 kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x)
                if batch_norm_state_hidden:
                    x = layers.normalization.BatchNormalization()(x)
                x = layers.LeakyReLU(alpha=0.1)(x)
    else:
        x = input_state_concat
    head_state = x

    # input_action
    input_action = layers.Input([n_actions])

    if 0<len(hidden_action_dims):
        for i, hidden_dim in enumerate(hidden_action_dims):
            if i == 0:
                x = layers.Dense(hidden_dim,
                                 kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(input_action)
                if batch_norm_action_input:
                    x = layers.normalization.BatchNormalization()(x)
                x = layers.LeakyReLU(alpha=0.1)(x)
            else:
                x = layers.Dense(hidden_dim,
                                 kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x)
                if batch_norm_action_hidden:
                    x = layers.normalization.BatchNormalization()(x)
                x = layers.LeakyReLU(alpha=0.1)(x)
    else:
        x = input_action

    if action_reg:
        x = layers.Lambda(lambda x: (x - action_min) / (action_max - action_min))(x)

    head_action = x

    # concat state, action and after
    input_concat = layers.concatenate([head_state, head_action], axis=-1)
    if 0<len(hidden_dims):
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                x = layers.Dense(hidden_dim,
                                 kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(input_concat)
                if batch_norm_hidden:
                    x = layers.normalization.BatchNormalization()(x)
                x = layers.LeakyReLU(alpha=0.1)(x)
            else:
                x = layers.Dense(hidden_dim,
                                 kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x)
                if batch_norm_hidden:
                    x = layers.normalization.BatchNormalization()(x)
                x = layers.LeakyReLU(alpha=0.1)(x)
    else:
        x = input_concat

    y = layers.Dense(1, activation='linear',
                     kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x)

    if value_reg:
        y = layers.Lambda(lambda x: (x - value_min) / (value_max - value_min))(y)

    return models.Model([*input_state, input_action], y)


########################################################################################################################
class Agent(object):
    def __init__(self, model_policy, model_value, val_min, val_max):
        self.model_policy = model_policy
        self.model_value = model_value
        self.val_min = val_min; self.val_max = val_max


    def get_action(self, state, sigma=1.0, greedy=False,
                   boltzmann=False, n_samples=128, tau=1.0, mode='rank',
                   get_log=False):
        '''
        出力タイプが8通りあるので注意
        greedy, [boltzmann, not boltzmann], not get_log -> 出力はgreedyでQ値は出力されない
        not greedy, not boltzmann, not get_log -> εグリーディでQ値は出力されない

        greedy, [boltzmann, not boltzmann], get_log -> 出力はgreedyだがQ値も出力される
        not greedy, not boltzmann, get_log -> εグリーディでQ値も出力される
        not greedy, boltzmann, get_log -> 出力はboltzmannでQ値も出力される
        not greedy, boltzmann, not get_log -> 上に同じだがQ値は出力されない

        要するに，boltzmannかget_logのいずれかがアクティブだとQ値を計算する必要がある．


        :param state:
        :param sigma:
        :param greedy:
        :param boltzmann:
        :param n_samples:
        :param tau:
        :param mode:
        :param get_log:
        :return:
        '''
        # sigma may be number or list (dim is equal to n_actions).
        mu = self.model_policy.predict_on_batch([*state])  # N, n_actions
        # print('mu = ', mu, 'Z = ',  Z, 'sigma = ', sigma)
        if boltzmann or get_log:
            Z = np.random.randn(n_samples * mu.shape[0], mu.shape[1])
            action_base = np.clip(np.tile(mu, (n_samples, 1)) + np.array(sigma).reshape(1, -1) * Z,
                                  self.val_min, self.val_max)

            Q = self.model_value.predict_on_batch([np.tile(S, (n_samples, 1)) for S in state] + [action_base, ]) \
                .reshape(n_samples, mu.shape[0]).T  # n_agents, n_samples

            if mode == 'rank':
                Q = np.array([Q.shape[1] - scipy.stats.mstats.rankdata(Q[a]) + 1 for a in range(Q.shape[0])])
                Q = Q / Q.shape[1]  # normalize by n_samples  ?

            Q = Q / (tau + 1e-6)
            Q = Q - np.max(Q, axis=-1, keepdims=True)
            Q = np.exp(Q)

            action_sample = action_base

        if greedy:  # greedy
            action = np.clip(mu, self.val_min, self.val_max)

        else:
            if not boltzmann:  # epsilon greedy
                Z = np.random.randn(*mu.shape)  # N, n_actions
                action = np.clip(mu + np.array(sigma).reshape(1, -1) * Z, self.val_min, self.val_max)

            else:  # boltzmann
                action_base = action_base.reshape(n_samples, mu.shape[0], mu.shape[1])
                action_prob = Q / np.sum(Q, axis=-1, keepdims=True)
                action = np.array([action_base[np.random.choice(np.arange(n_samples), p=action_prob[a]), a]
                                   for a in range(len(mu))])

        if get_log:
            return action, Q, action_sample
        else:
            return action


########################################################################################################################
class StratifiedExperienceBuffer(object):
    '''
    Stratified Experience Replay：層化経験再生
    ・バッファー自体は1つ，カテゴリを管理するリストを用いて，複数のクラスをカテゴリを同時に管理する．
    ・Trainerからappendされる際は基本的にそのまま足していく．
    ・容量を大幅に超過するか，明示的に呼び出された場合に経験を捨てる．
    ・外部からバッチを要求される場合のAPIは若干変える．

    Prioritized Experience Replay：優先度付経験再生


    '''
    def __init__(self, capacity=10**4, transfer_threshold=100, capacity_threshold=2*10**4,
                 gamma=1.0, n_advantages=4,
                 category_weights=None, default_weight=1.0):
        # buffer memory of main experience buffer
        self.state_current = []
        self.action = None
        self.reward = None
        self.state_next = []
        self.done = None
        self.pvf = None  # present value factor for n-step advantages when TD(n) calculation.
        self.category = None  # NDArray of string @ shape(N, 1)
        self.priority = None
        self.max_priority = 0

        # intermediate buffer before episode end and related
        self.int_buffer = []
        self.int_transferred = None  # 転送済みかどうかを保存するNDArray

        # parameters for mainly advantage
        self.gamma = gamma
        self.n_advantages = n_advantages

        # parameters for mainly stratify
        self.category_weights = category_weights
        self.default_weight = default_weight

        # parameters
        self.capacity = capacity
        self.transfer_threshold = transfer_threshold
        self.capacity_threshold = capacity_threshold

        self.computation_time = defaultdict(float)


    def reset(self):
        self.state_current = []
        self.action = None
        self.reward = None
        self.state_next = []
        self.done = None
        self.pvf = None
        self.category = None
        self.priority = None
        self.max_priority = 0

        self.int_buffer = []
        self.int_transferred = None


    def reset_int_buffer(self):  # should be called when evaluation.
        self.int_buffer = []
        self.int_transferred = None


    def get_batch(self, batch_size=256,
                  alpha=0.5, beta=0.5, mode='rank',
                  category_weights=None, default_weight=None):
        #
        if len(self) == 0:
            return None

        # priority
        priority = self.priority.ravel()
        if mode=='rank':
            priority = len(priority) - scipy.stats.mstats.rankdata(priority) + 1  # 最小値が0なので1を足す．
        else:
            if np.all(priority==0):
                priority = np.ones(priority.shape)
            else:
                priority = priority + 1e-6

        # alpha適用，標準化
        priority = priority / np.max(priority)
        priority = np.power(priority, alpha)
        priority = priority / np.sum(priority)

        # category_weight形成
        categories = set(self.category.ravel())

        if category_weights is None:
            category_weights = self.category_weights
        if default_weight is None:
            default_weight = self.default_weight

        if category_weights is None:
            category_weights = {category: 1 / len(categories)
                                for category in categories}
        else:
            category_weights = {category: category_weights[category] if category in category_weights.keys() else default_weight
                                for category in categories}
            category_weights = {category: category_weights[category] / sum(category_weights.values())
                                for category in categories}

        # idx形成
        idx = np.concatenate([np.random.choice(np.arange(len(self))[self.category.ravel()==category],
                                               p=priority[self.category.ravel()==category] / np.sum(priority[self.category.ravel()==category]),
                                               size=max(1, int(batch_size * category_weights[category])))
                              for category in categories])
        np.random.shuffle(idx)

        # 抽出
        state_current = [state[idx] for state in self.state_current]
        state_next = [state[idx] for state in self.state_next]
        action = self.action[idx]
        reward = self.reward[idx]
        done = self.done[idx]
        pvf = self.pvf[idx]

        # for PER.〜should be invesitigated.
        weight = np.power(batch_size * priority[idx], -beta)
        weight = weight / np.max(weight)

        return state_current, action, reward, state_next, done, pvf, weight, idx


    def update_priority(self, idx, priorities):
        # model側の計算結果に応じてpriorityを更新する．基本的には，get_batchメソッドとセットで使う想定になっている．
        self.priority[idx] = priorities
        self.max_priority = np.max(self.priority)


    def append(self, experience):  # int_bufferに追加するのみで，この時点では本体のexp_bufferには送らない．
        s_current, action, reward, s_next, done, category = experience
        self.int_buffer.append({
            'state_current': s_current,
            'action': action,
            'reward': reward,
            'state_next': s_next,
            'done': done,
            'category': category,
        })

        if self.transfer_threshold < len(self.int_buffer) or np.any(done):
            self.transfer_to_exp_buffer(gamma=self.gamma, n_advantages=self.n_advantages)


    def refine_to_capacity(self, category_weights=None, default_weight=1.0):
        # capacity内に収めるように古い経験を削除する．但し，stratifyを可能な限り維持する．
        # 恐らくそれなりに重いので，頻繁には呼び出さないこと．
        start_time = time.time()
        if len(self) == 0:
            return None

        # 各カテゴリの重みを取得
        categories = set(self.category.ravel())
        if category_weights is None:
            category_weights = {category: 1 / len(categories)
                                for category in categories}
        else:
            category_weights = {category: category_weights[category] if category in category_weights.keys() else default_weight
                                for category in categories}
            category_weights = {category: category_weights[category] / sum(category_weights.values())
                                for category in categories}

        # capacity内に収めるようにした場合に各カテゴリ内の個数が幾つになるべきか
        category_counts = {category: max(1, int(self.capacity * category_weights[category]))
                           for category in categories}

        # 抽出
        for category in categories:
            x = np.arange(len(self))[self.category.ravel()==category][-category_counts[category]:]
        idx = np.concatenate([np.arange(len(self))[self.category.ravel()==category][-category_counts[category]:]
                             for category in categories])
        self.state_current = [state[idx] for state in self.state_current]
        self.state_next = [state[idx] for state in self.state_next]
        self.action = self.action[idx]
        self.reward = self.reward[idx]
        self.done = self.done[idx]
        self.pvf = self.pvf[idx]
        self.category = self.category[idx]
        self.priority = self.priority[idx]
        self.max_priority = np.max(self.priority)

        self.computation_time['refine_to_capacity'] += time.time() - start_time;


    def transfer_to_exp_buffer(self, gamma=0.9, n_advantages=4):
        if len(self.int_buffer)==0:
            return None

        done = np.concatenate([int_buffer['done'].T for int_buffer in self.int_buffer], axis=0).astype(bool)  # n_agents, 1 -> T, n_agents

        # tの終端側から，doneが始まる範囲・全てのcategoryが確定する範囲，0の順序に並んでいる．
        # 最後尾のdoneの位置を取得．
        t_done = np.any(done, axis=1)  # T
        t_done = -1 if np.any(t_done)==False else np.where(t_done)[0][-1]
        if t_done==-1:
            return None
        n_agents = done.shape[1]  # n_agents, 1

        # 転送管理配列作成
        start_time = time.time()
        if self.int_transferred is None:
            self.int_transferred = np.full(shape=(t_done+1, n_agents), fill_value=False, dtype=bool)
        else:
            if t_done+1 - len(self.int_transferred) == 0:  # 前回実行時から実質的にデータが増えていない．
                return None
            else:
                self.int_transferred = np.concatenate([self.int_transferred,  # 前回までの部分
                                                       np.full(shape=(t_done+1 - len(self.int_transferred), n_agents), fill_value=False, dtype=bool)], axis=0)
        self.computation_time['転送管理配列作成'] += time.time() - start_time; start_time = time.time()

        # categoryの付与
        int_category = np.full(shape=(t_done + 1, n_agents), fill_value=None)
        int_category[-1][done[t_done]] = self.int_buffer[t_done]['category'][done[t_done]].ravel()
        for t in range(t_done-1, 0-1, -1):
            int_category[t] = int_category[t + 1]
            int_category[t][done[t]] = self.int_buffer[t]['category'][done[t]].ravel()
        self.computation_time['カテゴリ付与'] += time.time() - start_time; start_time = time.time()

        # 転送用マスク&インデックス作成
        transfer_mask = np.logical_and(int_category != None, np.logical_not(self.int_transferred))  # t, n_agents
        transfer_idx = np.where(transfer_mask)
        self.computation_time['転送用マスクとインデックス作成'] += time.time() - start_time; start_time = time.time()

        # exp_bufferに転送
        R = np.array([self.int_buffer[t]['reward'] for t in range(t_done + 1)])[:, :, 0]
        reward_cum = np.array([np.sum(np.concatenate([np.ones(shape=(1, n_agents)),
                                             np.cumprod((1-done[t: min(t + n_advantages, t_done + 1)][: -1]) * gamma, axis=0)], axis=0)
                        * R[t: t + n_advantages], axis=0)
                 for t in range(t_done+1)])
        n_cum = np.array([np.sum(np.cumprod((1-done[t: min(t + n_advantages, t_done + 1) - 1]), axis=0), axis=0)
                 for t in range(t_done+1)])
        state_next_idx = np.arange(t_done+1).reshape(-1, 1) + n_cum
        self.computation_time['exp_bufferに転送用データ作成：序盤'] += time.time() - start_time; start_time = time.time()

        s_current = [np.array([self.int_buffer[t]['state_current'][s][a] for t, a in zip(*transfer_idx)])
                     for s in range(len(self.int_buffer[0]['state_current']))]
        s_next = [np.array([self.int_buffer[state_next_idx[t, a]]['state_next'][s][a] for t, a in zip(*transfer_idx)])
                     for s in range(len(self.int_buffer[0]['state_next']))]
        self.computation_time['exp_bufferに転送用データ作成：中盤'] += time.time() - start_time; start_time = time.time()

        action = np.array([self.int_buffer[t]['action'][a] for t, a in zip(*transfer_idx)])  # different from DQN_SER
        reward_cum = np.array([reward_cum[t][a] for t, a in zip(*transfer_idx)]).reshape(-1, 1)
        done = np.array([done[state_next_idx[t, a], a] for t, a in zip(*transfer_idx)]).reshape(-1, 1)
        pvf = gamma ** (1 + np.array([n_cum[t][a] for t, a in zip(*transfer_idx)])).reshape(-1, 1)
        category = np.array([int_category[t, a] for t, a in zip(*transfer_idx)]).reshape(-1, 1)
        priority = np.full_like(done, self.max_priority)
        self.computation_time['exp_bufferに転送用データ作成：終盤'] += time.time() - start_time; start_time = time.time()

        #　全エージェントがtransferされているデータを削除：これは，全てのdataにcategoryが付与された部分である．
        self.int_transferred = np.logical_or(self.int_transferred, transfer_mask)
        t_transferred = np.all(self.int_transferred, axis=1)
        t_transferred = -1 if any(t_transferred)==False else np.where(t_transferred)[0][-1]
        if t_transferred != -1:
            self.int_buffer = self.int_buffer[t_transferred+1:]
            self.int_transferred = self.int_transferred[t_transferred+1:]
        self.computation_time['int_bufferリフレッシュ'] += time.time() - start_time; start_time = time.time()

        # 初回のみの処理
        if len(self)==0:
            self.state_current = [state_ex for state_ex in s_current]
            self.state_next = [state_ex for state_ex in s_next]
            self.action = action
            self.reward = reward_cum
            self.done = done
            self.pvf = pvf
            self.category = category
            self.priority = np.full_like(done, self.max_priority)
            return None

        self.state_current = [np.concatenate([state_in, state_ex], axis=0)
                              for state_in, state_ex in zip(self.state_current, s_current)]
        self.state_next = [np.concatenate([state_in, state_ex], axis=0)
                           for state_in, state_ex in zip(self.state_next, s_next)]
        self.action = np.concatenate([self.action, action], axis=0)
        self.reward = np.concatenate([self.reward, reward_cum], axis=0)
        self.done = np.concatenate([self.done, done], axis=0)
        self.pvf = np.concatenate([self.pvf, pvf], axis=0)
        self.category = np.concatenate([self.category, category], axis=0)
        self.priority = np.concatenate([self.priority, priority], axis=0)

        if self.capacity_threshold < len(self):
            self.refine_to_capacity(category_weights=self.category_weights, default_weight=self.default_weight)
        self.computation_time['最終転送処理'] += time.time() - start_time; start_time = time.time()


    def __len__(self):
        if len(self.state_current) == 0:
            return 0
        else:
            return len(self.state_current[0])


########################################################################################################################
class Trainer(object):
    def __init__(self, env, agent, exp_buffer=None):
        self.env = env
        self.agent = agent
        self.model_policy = agent.model_policy
        self.model_value = agent.model_value
        self.exp_buffer = exp_buffer if exp_buffer is not None else StratifiedExperienceBuffer()

        # generate target model for policy and value.
        self.model_policy_target = models.Model.from_config(self.model_policy.get_config())
        self.model_value_target = models.Model.from_config(self.model_value.get_config())
        self.model_policy_target.set_weights(self.model_policy.get_weights())
        self.model_value_target.set_weights(self.model_value.get_weights())

        self.loss_history = {'all':[], 'value':[], 'pv':[]}  # pv means policy value, but substantially, policy.
        self.return_history = {'all':[]}
        self.loss_color = {'all':'r', 'value':'g', 'pv':'b'}  # for graph
        self.return_color = {'all':'b'}  # for graph

        self.log_exp = {'state_current': [],
                        'action': [],
                        'reward': [],
                        'state_next': [],
                        'done': [],
                        'info': [],
                        'Q': [],
                        'action_sample': []}
        self.log_eva = {'state_current': [],
                        'action': [],
                        'reward': [],
                        'state_next': [],
                        'done': [],
                        'info': [],
                        'Q': [],
                        'action_sample': []}


    def train(self, n_steps=10000, training_interval=100, n_batches=10, batch_size=256,
              target_update_interval_policy=100, target_update_interval_value=100,
              tau_policy=1.0, tau_value=1.0,
              buf_alpha=0.5, buf_beta=0.5, buf_mode='rank',
              gamma=1.0,
              optimizer_pv=optimizers.Adam(1e-2), optimizer_value=optimizers.Adam(1e-2),
              sigma_start=1.0, sigma_end=0.1, sigma_interval=10000,
              boltzmann=False, tau_start=1.0, tau_end=1.0, tau_interval=10000, n_samples=128,
              policy_sampling_update=False, policy_sampling_n=64,
              policy_sampling_sigma_start=10.0, policy_sampling_sigma_end=1.0, policy_sampling_sigma_interval=10000,
              warmpup_steps=500,
              n_advantages=10, category_weights=None, default_weight=1.0,
              verbose=True, verbose_interval=10, evaluate_interval=100,
              get_log=False):

        # generate model
        self.model_value.compile(optimizer=optimizer_value, loss='mse')
        self.model_value.trainable = False
        model_pv_output = self.model_value([*self.model_value.inputs][: -1]
                                           +[self.model_policy([*self.model_value.inputs][: -1]), ])
        self.model_pv = models.Model([*self.model_value.inputs][: -1], model_pv_output)
        self.model_pv.compile(optimizer=optimizer_pv, loss=self.loss_pv)
        self.model_policy.compile(optimizer=optimizer_pv, loss='mse')  # added for sample updater.

        # setting exp_buffer
        self.exp_buffer.reset()
        self.exp_buffer.gamma = gamma
        self.exp_buffer.n_advantages = n_advantages
        self.exp_buffer.category_weights = category_weights
        self.exp_buffer.default_weight = default_weight

        self.reset_history()
        self.reset_log()
        loss = None

        for t in tqdm(range(n_steps)):
            # interaction with environment
            state_current = self.env.state()
            sigma = np.clip(sigma_start + (sigma_end - sigma_start) / sigma_interval * t,
                            np.minimum(sigma_start, sigma_end), np.maximum(sigma_start, sigma_end)).ravel()[0]
            tau = np.clip(tau_start + (tau_end - tau_start) / tau_interval * t,
                          min(tau_start, tau_end), max(tau_start, tau_end)).ravel()[0]
            if get_log:
                action, Q, action_sample = self.agent.get_action(state_current, sigma, False, boltzmann, n_samples, tau, get_log=get_log)
            else:
                action = self.agent.get_action(state_current, sigma, False, boltzmann, n_samples, tau, get_log=get_log)
            state_next, reward, info, done = self.env.step(action)
            self.exp_buffer.append([state_current, action, reward, state_next, done, info])

            # update base models through NN learning
            if (t+1) % training_interval == 0 and warmpup_steps < t:
                if policy_sampling_update:
                    policy_sampling_sigma = np.clip(
                        policy_sampling_sigma_start + (policy_sampling_sigma_start - policy_sampling_sigma_end) / policy_sampling_sigma_interval * t,
                        np.minimum(policy_sampling_sigma_start, policy_sampling_sigma_end),
                        np.maximum(policy_sampling_sigma_start, policy_sampling_sigma_end)
                    ).ravel()[0]
                else:
                    policy_sampling_sigma = 0

                loss = self._train_on_batch(n_batches, batch_size,
                                            buf_alpha, buf_beta, buf_mode,
                                            policy_sampling_update, policy_sampling_n, policy_sampling_sigma)
                self.loss_history['all'].append(loss[0] + loss[1])  # total loss
                self.loss_history['pv'].append(loss[0])  # loss of policy value network
                self.loss_history['value'].append(loss[1])  # loss of value network

            # update model_target_policy
            if (t+1) % target_update_interval_policy == 0:
                weights_base = self.model_policy.get_weights()
                weights_target = self.model_policy.get_weights()
                new_weights = [(weight_base * tau_policy + weight_target * (1 - tau_policy))
                               for weight_base, weight_target in zip(weights_base, weights_target)]
                self.model_policy_target.set_weights(new_weights)

            # update model_target_value
            if (t+1) % target_update_interval_value == 0:
                weights_base = self.model_value.get_weights()
                weights_target = self.model_value_target.get_weights()
                new_weights = [(weight_base * tau_value + weight_target * (1 - tau_value))
                               for weight_base, weight_target in zip(weights_base, weights_target)]
                self.model_value_target.set_weights(new_weights)

            # log
            if get_log:
                self.log_exp['state_current'].append(state_current)
                self.log_exp['action'].append(action)
                self.log_exp['reward'].append(reward)
                self.log_exp['state_next'].append(state_next)
                self.log_exp['done'].append(done)
                self.log_exp['info'].append(info)
                self.log_exp['Q'].append(Q)
                self.log_exp['action_sample'].append(action_sample)

            # verbose
            if verbose and (t+1) % verbose_interval == 0 and loss is not None:
                print('step = {}, loss_all = {:.5f}, loss_pv = {:.5f}, loss_value = {:.5f}'
                      .format(t, loss[0]+loss[1], loss[0], loss[1]))

            # evaluate
            if (t+1) % evaluate_interval == 0:
                self.return_history['all'].append(self.evaluate(show_log=verbose, gamma=gamma,
                                                             get_log=get_log, sigma=sigma))

        # lossをNDArrayに変換
        self.loss_history = {key: np.array(self.loss_history[key]) for key in self.loss_history.keys()}
        self.return_history = {key: np.array(self.return_history[key]) for key in self.return_history.keys()}

        # log統合
        if get_log:
            for key in ['state_current', 'state_next']:
                pass
                self.log_exp[key] = [np.concatenate(S, axis=0) for S in zip(*self.log_exp[key])]
                self.log_eva[key] = [np.concatenate(S, axis=0) for S in zip(*self.log_eva[key])]

            for key in ['action', 'reward', 'done', 'info', 'Q', 'action_sample']:
                self.log_exp[key] = np.concatenate(self.log_exp[key], axis=0)
                self.log_eva[key] = np.concatenate(self.log_eva[key], axis=0)


    def reset_history(self):
        self.loss_history = {key: [] for key in self.loss_history.keys()}
        self.return_history = {key: [] for key in self.return_history.keys()}


    def reset_log(self):
        self.log_exp = {key: [] for key in self.log_exp.keys()}
        self.log_eva = {key: [] for key in self.log_eva.keys()}


    def _train_on_batch(self, n_batches, batch_size,
                        buf_alpha, buf_beta, buf_mode,
                        policy_sampling_update=True, n_samples=64, sigma=10.0):
        if len(self.exp_buffer) == 0:
            return None

        losses_pv = []
        losses_value = []
        for i in range(n_batches):
            S_c, A_c, R, S_n, done, pvf, weight, idx = \
                self.exp_buffer.get_batch(batch_size, buf_alpha, buf_beta, buf_mode)

            # target生成
            A_n = self.model_policy_target.predict([*S_n])
            Q_pred_next = self.model_value_target.predict_on_batch([*S_n] + [A_n, ])
            Q_pred_curr = self.model_value.predict_on_batch([*S_c] + [A_c, ])  # used in PER update.
            Q_pred_next[done] = 0.0
            target = R.ravel() + pvf.ravel() * Q_pred_next.ravel()

            # update model_value
            with warnings.catch_warnings():  # suepress warnings by keras
                warnings.simplefilter('ignore')
                loss_value = self.model_value.train_on_batch([*S_c] + [A_c, ], target)

            # update model_policy through model_pv or sampling method.
            if not policy_sampling_update:
                loss_pv = self.model_pv.train_on_batch([*S_c], target)  # target is dummy.
            else:
                # A_c
                A_base = self.model_policy.predict_on_batch([*S_c])  # batch_size, n_actions
                Z = np.random.randn(n_samples * A_base.shape[0], A_base.shape[1])
                A_sample = np.tile(A_base, (n_samples, 1)) + sigma * Z
                Q_sample = self.model_value.predict_on_batch(
                    [np.tile(S, (n_samples, 1)) for S in S_c] + [A_sample, ])\
                    .reshape(n_samples, A_base.shape[0]).T  # batch_size, n_samples
                A_sample = A_sample.reshape(n_samples, A_base.shape[0], A_base.shape[1])
                # NDArrayの軸を入れ替える処理が必要．
                # A_sample = np.transpose(A_sample, [1, 2, 0])
                A_target = A_sample[np.argmax(Q_sample, axis=1), np.arange(A_base.shape[0])]

                loss_pv = self.model_policy.train_on_batch([*S_c], A_target)

            losses_pv.append(loss_pv)
            losses_value.append(loss_value)

            # exp_bufferを更新：PER
            priority = np.abs(target - Q_pred_curr.ravel()).reshape(-1, 1)  # different from DQN_SER
            self.exp_buffer.update_priority(idx, priority)

        # convert losses to NDArray
        losses_pv = np.array(losses_pv).mean()
        losses_value = np.array(losses_value).mean()

        return losses_pv, losses_value


    def loss_pv(self, y_true, y_pred):
        return -y_pred


    def reset_history(self):
        self.loss_history_all = []
        self.loss_history_pv = []
        self.loss_history_value = []
        self.reward_history_all = []


    def evaluate(self, show_log=False, time_horizon=1000, n_showing_agents=10, gamma=1.0,
                 get_log=False, sigma=1.0):
        self.env.reset()
        rewards = []
        done_mask = 1
        for t in range(time_horizon):
            state_current = self.env.state()
            if get_log:
                action, Q, action_sample = self.agent.get_action(state_current, sigma=sigma, greedy=True, get_log=get_log)
            else:
                action = self.agent.get_action(state_current, greedy=True, get_log=get_log)
            state_next, reward, info, done = self.env.step(action)
            rewards.append(done_mask * reward)  # apply done mask for cut off already done agents
            done_mask = done_mask * (1-np.array(done, dtype=int)) * gamma  # update done_mask

            # log
            if get_log:
                self.log_eva['state_current'].append(state_current)
                self.log_eva['action'].append(action)
                self.log_eva['reward'].append(reward)
                self.log_eva['state_next'].append(state_next)
                self.log_eva['done'].append(done)
                self.log_eva['info'].append(info)
                self.log_eva['Q'].append(Q)
                self.log_eva['action_sample'].append(action_sample)

            if show_log:
                print('action = ', action.ravel()[: n_showing_agents])

            if done_mask.sum() == 0:  # end if all agents are done
                break

        rewards = np.array(rewards)[:, :, 0]

        if show_log:
            print('return', rewards.sum(axis=0)[: n_showing_agents])

        return np.mean(rewards.sum(axis=0))


########################################################################################################################
class Tester(object):
    def __init__(self, env, model_policy, model_value, val_min, val_max, name='tester_DDPG_SER'):
        self.env = env
        self.model_policy = model_policy
        self.model_value = model_value
        self.agent = Agent(model_policy, model_value, val_min, val_max)
        self.trainer = Trainer(env, self.agent)

        self.name = name


    def test(self,
             n_trials=5,
             n_steps=10000, training_interval=100, n_batches=10, batch_size=256,
             target_update_interval_policy=100, target_update_interval_value=100,
             tau_policy=1.0, tau_value=1.0,
             buf_alpha=0.5, buf_beta=0.5, buf_mode='rank',
             gamma=1.0,
             optimizer_pv=optimizers.Adam(1e-2), optimizer_value=optimizers.Adam(1e-2),
             sigma_start=1.0, sigma_end=0.1, sigma_interval=10000,
             boltzmann=False, tau_start=1.0, tau_end=1.0, tau_interval=10000, n_samples=128,
             policy_sampling_update=False, policy_sampling_n=64,
             policy_sampling_sigma_start=10.0, policy_sampling_sigma_end=1.0, policy_sampling_sigma_interval=10000,
             warmpup_steps=500,
             n_advantages=10, category_weights=None, default_weight=1.0,
             verbose=True, verbose_interval=10, evaluate_interval=100,
             get_log=False, save_objects=False
             ):
        print('start testing... : ', datetime.now())
        self.loss_histories = []
        self.return_histories = []
        for trial in range(n_trials):
            print('start {}/{} th trial...'.format(trial, n_trials))
            self.trial(
                n_steps=n_steps, training_interval=training_interval, n_batches=n_batches, batch_size=batch_size,
                target_update_interval_policy=target_update_interval_policy,
                target_update_interval_value=target_update_interval_value,
                tau_policy=tau_policy, tau_value=tau_value,
                buf_alpha=buf_alpha, buf_beta=buf_beta, buf_mode=buf_mode,
                gamma=gamma,
                optimizer_pv=optimizer_pv, optimizer_value=optimizer_value,
                sigma_start=sigma_start, sigma_end=sigma_end, sigma_interval=sigma_interval,
                boltzmann=boltzmann, tau_start=tau_start, tau_end=tau_end, tau_interval=tau_interval,
                n_samples=n_samples,
                policy_sampling_update=policy_sampling_update,
                policy_sampling_n=policy_sampling_n,
                policy_sampling_sigma_start=policy_sampling_sigma_start,
                policy_sampling_sigma_end=policy_sampling_sigma_end,
                policy_sampling_sigma_interval=policy_sampling_sigma_interval,
                warmpup_steps=warmpup_steps,
                n_advantages=n_advantages, category_weights=category_weights, default_weight=default_weight,
                verbose=verbose, verbose_interval=verbose_interval, evaluate_interval=evaluate_interval,
                get_log=get_log
            )

        print('end testing... : ', datetime.now())
        self.report(gamma=gamma)

        # saving
        with open(str(self.name) + '.pkl', 'wb') as f:
            if save_objects:
                pickle.dump([self.trainer , self.loss_histories, self.return_histories], f)
            else:
                pickle.dump([self.loss_histories, self.return_histories], f)


    def trial(self,
              n_steps, training_interval, n_batches, batch_size,
              target_update_interval_policy, target_update_interval_value,
              tau_policy, tau_value,
              buf_alpha, buf_beta, buf_mode,
              gamma,
              optimizer_pv, optimizer_value,
              sigma_start, sigma_end, sigma_interval,
              boltzmann, tau_start, tau_end, tau_interval, n_samples,
              policy_sampling_update, policy_sampling_n,
              policy_sampling_sigma_start, policy_sampling_sigma_end, policy_sampling_sigma_interval,
              warmpup_steps,
              n_advantages, category_weights, default_weight,
              verbose, verbose_interval, evaluate_interval,
              get_log
              ):

        # reset models
        self.model_policy = models.Model.from_config(self.model_policy.get_config())
        self.model_value = models.Model.from_config(self.model_value.get_config())
        self.agent = Agent(self.model_policy, self.model_value,
                           self.agent.val_min, self.agent.val_max)
        self.trainer = Trainer(self.env, self.agent)

        # reset optimizers
        optimizer_pv = optimizer_pv.from_config(optimizer_pv.get_config())
        optimizer_value = optimizer_value.from_config(optimizer_value.get_config())

        # 学習
        self.trainer.train(
            n_steps=n_steps, training_interval=training_interval, n_batches=n_batches, batch_size=batch_size,
            target_update_interval_policy=target_update_interval_policy,
            target_update_interval_value=target_update_interval_value,
            tau_policy=tau_policy, tau_value=tau_value,
            buf_alpha=buf_alpha, buf_beta=buf_beta, buf_mode=buf_mode,
            gamma=gamma,
            optimizer_pv=optimizer_pv, optimizer_value=optimizer_value,
            sigma_start=sigma_start, sigma_end=sigma_end, sigma_interval=sigma_interval,
            boltzmann=boltzmann, tau_start=tau_start, tau_end=tau_end, tau_interval=tau_interval, n_samples=n_samples,
            policy_sampling_update=policy_sampling_update,
            policy_sampling_n=policy_sampling_n,
            policy_sampling_sigma_start=policy_sampling_sigma_start,
            policy_sampling_sigma_end=policy_sampling_sigma_end,
            policy_sampling_sigma_interval=policy_sampling_sigma_interval,
            warmpup_steps=warmpup_steps,
            n_advantages=n_advantages, category_weights=category_weights, default_weight=default_weight,
            verbose=verbose, verbose_interval=verbose_interval, evaluate_interval=evaluate_interval,
            get_log=get_log,
        )

        # ログ取得
        self.loss_histories.append(self.trainer.loss_history)
        self.return_histories.append(self.trainer.return_history)


    ###################################################################################
    # below codes are for reporting.
    def report(self, loss_reward=True, action=True, param=True, gamma=1.0):
        if loss_reward:
            self.rep_loss_reward()
        if action:
            self.rep_action(gamma=gamma)
        if param:
            self.rep_param()


    def rep_loss_reward(self, figsize=(15, 7), alpha=None, fontsize=14, dpi=100):
        print('loss and reward history')

        if alpha is None:
            alpha = np.sqrt(1 / len(self.loss_histories))

        fig = plt.figure(figsize=figsize, dpi=dpi)

        ax = fig.add_subplot(1, 2, 1)
        for i, loss_history in enumerate(self.loss_histories):
            for key in loss_history.keys():
                ax.plot(loss_history[key], c=self.trainer.loss_color[key], alpha=alpha, label=key if i == 0 else None)  # black
        ax.set_title('loss history', fontsize=fontsize)
        ax.legend(fontsize=fontsize)

        ax = fig.add_subplot(1, 2, 2)
        for i, return_history in enumerate(self.return_histories):
            for key in return_history.keys():
                ax.plot(return_history[key], c=self.trainer.return_color[key], alpha=alpha, label=key if i == 0 else None)  # black
        ax.set_title('return history', fontsize=fontsize)
        ax.legend(fontsize=fontsize)

        plt.tight_layout()
        plt.show()


    def rep_action(self, gamma=1.0):
        print('acquired action')
        self.trainer.evaluate(show_log=True, n_showing_agents=10, gamma=gamma)


    def rep_param(self, figsize=(15, 5), dpi=100):
        print('param distribution in model_policy')
        p_layers = [layer for layer in self.model_policy.layers if len(layer.get_weights()) != 0]

        fig = plt.figure(figsize=figsize, dpi=dpi)
        for i, layer in enumerate(p_layers):
            ax1 = fig.add_subplot(2, len(p_layers), 1+i)
            ax1.hist(layer.get_weights()[0].ravel(), bins=20)
            ax1.set_title(layer.name + ' : W')

            ax2 = fig.add_subplot(2, len(p_layers), 1+len(p_layers)+i)
            ax2.hist(layer.get_weights()[1].ravel(), bins=20)
            ax2.set_title(layer.name + ' : b')
        plt.tight_layout()
        plt.show()

        print('param distribution in model_value')
        p_layers = [layer for layer in self.model_value.layers if len(layer.get_weights())!=0]
        fig = plt.figure(figsize=figsize, dpi=dpi)
        for i, layer in enumerate(p_layers):
            ax1 = fig.add_subplot(2, len(p_layers), 1+i)
            ax1.hist(layer.get_weights()[0].ravel(), bins=20)
            ax1.set_title(layer.name + ' : W')

            ax2 = fig.add_subplot(2, len(p_layers), 1+len(p_layers)+i)
            ax2.hist(layer.get_weights()[1].ravel(), bins=20)
            ax2.set_title(layer.name + ' : b')
        plt.tight_layout()
        plt.show()


########################################################################################################################
class Visualizer(object):
    '''
    Visualizer of action and Q history.
    This class is specific to this DQN agent, not available for other agents.
    '''
    def __init__(self):
        pass


    def viz_action_history(self, trainer, state_valuation, action_valuation, data_span=10,
                           figsize=(15, 7), dpi=150, cmap='jet', fontsize=16, alpha=0.1, s=0.1):
        '''

        :param trainer: Trainer instance contains log
        :param state_valuation: function that maps state to number
        :return:
        '''
        def log_get(log):
            state_current = state_valuation(log['state_current'])
            action = action_valuation(log['action'])

            state_current = np.array(state_current).ravel()[::data_span]
            action = np.array(action).ravel()[::data_span]
            x = np.arange(len(state_current))

            return x, state_current, action

        fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
        x, t, A = log_get(trainer.log_exp)
        axs[0, 0].scatter(x=x, y=A, c=t, cmap=cmap, alpha=alpha, s=s)
        axs[0, 0].set_title('action : exploration', size=fontsize)
        axs[0, 0].set_xlabel('data step', size=fontsize)
        axs[0, 0].set_ylabel('action', size=fontsize)

        axs[1, 0].hist(A, bins=20)
        axs[1, 0].set_title('action histogram : exploration', size=fontsize)
        axs[1, 0].set_xlabel('action', size=fontsize)

        x, t, A = log_get(trainer.log_eva)
        axs[0, 1].scatter(x=x, y=A, c=t, cmap=cmap, alpha=alpha, s=s)
        axs[0, 1].set_title('action : evaluation', size=fontsize)
        axs[0, 1].set_xlabel('data step', size=fontsize)
        axs[0, 1].set_ylabel('action', size=fontsize)

        axs[1, 1].hist(A, bins=20)
        axs[1, 1].set_title('action histogram : evaluation', size=fontsize)
        axs[1, 1].set_xlabel('action', size=fontsize)

        plt.tight_layout()
        plt.show()


    def viz_Q_history(self, trainer,
                      state_valuation=lambda x: np.ravel(x[0][:, 0]),
                      action_valuation=lambda x: np.ravel(x[:, 0]),
                      data_span_exp=100, data_span_eva=100,
                      figsize=(15, 5), dpi=150, cmap='jet', fontsize=16):
        def QSA(log, data_span):
            Q = np.ravel(log['Q'])
            S = state_valuation(log['state_current'])
            S = np.tile(S.reshape(-1, 1), (1, len(Q) // len(S))).ravel()
            A = action_valuation(log['action_sample'])

            return Q[::data_span], S[::data_span], A[::data_span]

        plt.figure(figsize=figsize, dpi=dpi)
        Q, S, A = QSA(trainer.log_exp, data_span_exp)
        plt.scatter(x=np.arange(len(Q)), y=A, c=Q, s=1.0, alpha=1.0, cmap='jet')
        plt.title('evolution of Q in exploration', size=fontsize)
        plt.xlabel('data step', size=fontsize)
        plt.ylabel('action', size=fontsize)
        plt.show()

        plt.figure(figsize=figsize, dpi=dpi)
        Q, S, A = QSA(trainer.log_eva, data_span_eva)
        plt.scatter(x=np.arange(len(Q)), y=A, c=Q, s=1.0, alpha=1.0, cmap='jet')
        plt.title('evolution of Q in evaluation', size=fontsize)
        plt.xlabel('data step', size=fontsize)
        plt.ylabel('action', size=fontsize)
        plt.show()


########################################################################################################################
if __name__ == '__main__':
    n_agents = 4
    state_dims = [1, 3]
    n_actions = 10
    action_min = -0.1
    action_max = 1.1

    env = Environment(n_agents=n_agents, state_dims=state_dims, n_actions=n_actions)

    model_policy = gen_model_policy(
        state_shapes=[[d, ] for d in state_dims],
        n_actions=n_actions,
        hidden_dims=[2, 4],
        reg_l1=0.01, reg_l2=0.02,
        action_min=action_min, action_max=action_max,
    )

    model_value = gen_model_value(
        state_shapes=[[d, ] for d in state_dims],
        n_actions=n_actions,
        hidden_action_dims=[10, 5], hidden_state_dims=[3, 3], hidden_dims=[3, 3],
        reg_l1=0.01, reg_l2=0.01
    )

    start_time = time.time()
    trainer = Trainer(env, Agent(model_policy, model_value, val_min=0.0, val_max=10.0))
    trainer.train(verbose=False, gamma=0.99)
    print(time.time() - start_time)


    print('END')


