import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras import layers
from keras import regularizers
from keras import models
from keras import optimizers
import keras.backend as K

import scipy.stats  # used in exp_buffer for PER.
from collections import defaultdict


import math
from datetime import datetime
import time
from tqdm import tqdm
import pickle
import warnings  # for trainer for surpress keras warnings;




'''
AWR
https://xbpeng.github.io/projects/AWR/2019_AWR.pdf
1: initialize policy
2: reset experience replay buffer D
3: for iteration k=1 to kmax:
4:      add trajectories sampling from policy to D
5:      fit Value net V(s) as to minimize |R - V| in D
6:      update policy to log policy(a|s) exp((R-V)/beta) as to gradient ascend.

As a nature of this method, exp buffer must have episodic control.  


'''



########################################################################################################################
class Environment(object):
    def __init__(self, n_agents=2, state_dims=[1, 3, 5], n_outputs=5):
        self.n_agents = n_agents
        self.state_dims = state_dims
        self.n_outputs = n_outputs
        self.reset()


    def step(self, action):
        for state in self.states:
            state += (action==1) * np.random.randint(0, 2+1, size=(state.shape))
        reward = np.random.binomial(1, 0.2, self.done.shape)

        return self.state(), np.copy(reward), np.tile(np.array('no_info'), (self.n_agents, 1)), np.copy(self.done)


    def state(self):
        return [np.copy(state) for state in self.states]


    def reset(self):
        self.states = []
        for state_dim in self.state_dims:
            self.states.append(np.arange(state_dim) * np.ones(self.n_agents).reshape(-1, 1))
        self.done = np.ones(shape=(self.n_agents, 1), dtype=bool)


    def shapes(self):
        pass


########################################################################################################################
def gen_model(input_shapes=[[10], [20]], n_outputs=3,
              hidden_dims=[2, 4, 6], reg_l1=0.0, reg_l2=0.0,
              state_reg=False, state_min=-1.0, state_max=1.0,
              action_reg=False, action_min=-1.0, action_max=1.0,
              action_sigma_min=0.0, action_sigma_max=10.0,
              value_reg=False, value_min=-1.0, value_max=1.0):
    '''

    :param input_shapes:
    :param n_outputs:
    :param hidden_dims:
    :param reg_l1:
    :param reg_l2:
    :param state_reg:
    :param state_min:
    :param state_max:
    :param action_reg:
    :param action_min:
    :param action_max:
    :param value_reg:
    :param value_min:
    :param value_max:
    :return:
    '''
    input_ts = [layers.Input(input_shape) for input_shape in input_shapes]
    if 2 <= len(input_ts):
        input_concat = layers.concatenate(input_ts, axis=-1)
    else:
        input_concat = layers.Lambda(lambda x: x)(input_ts[0])  # 恒等レイヤー

    if state_reg:
        input_concat = layers.Lambda(lambda x: (x - state_min) / (state_max - state_min))(input_concat)

    for i, hidden_dim in enumerate(hidden_dims):
        if i == 0:
            x = layers.LeakyReLU(alpha=0.1)(layers.Dense(hidden_dim, kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(input_concat))
        else:
            x = layers.LeakyReLU(alpha=0.1)(layers.Dense(hidden_dim, kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x))

    if action_reg:
        policy_mu_head = layers.Dense(n_outputs, activation='sigmoid',
                                   kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x)
        policy_mu_head = layers.Lambda(lambda x: action_min + (action_max - action_min) * x)(policy_mu_head)
    else:
        policy_mu_head = layers.Dense(n_outputs, activation='linear',
                                   kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x)

    policy_sd_head = layers.Dense(n_outputs, activation='sigmoid',
                                  kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x)
    policy_sd_head = layers.Lambda(lambda x: action_sigma_min + (action_sigma_max - action_sigma_min) * x)(policy_sd_head)

    policy_head = layers.concatenate([policy_mu_head, policy_sd_head], axis=-1, name='policy_head')

    value_head = layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x)
    if value_reg:
        value_head = layers.Lambda(lambda x: value_min + (value_max - value_min) * x)(value_head)

    return models.Model(input_ts, [policy_head, value_head])


########################################################################################################################
class Agent(object):
    def __init__(self, model, action_min=-np.inf, action_max=np.inf):
        self.model = model
        self.action_min = action_min
        self.action_max = action_max


    def get_action(self, state, greedy=False, get_log=False):
        ps, value = self.model.predict_on_batch([*state])
        mu = ps[:, :ps.shape[-1] // 2]; sd = ps[:, ps.shape[-1] // 2:]  # N, n_actions
        Z = np.random.randn(*mu.shape) if not greedy else 0.0
        if get_log:
            return np.clip(mu + sd * Z, self.action_min, self.action_max), mu, sd, value
        else:
            return np.clip(mu + sd * Z, self.action_min, self.action_max)


########################################################################################################################
class StratifiedExperienceBuffer(object):  # No changes from that of DDPG
    '''
    Stratified Experience Replay：層化経験再生
    ・バッファー自体は1つ，カテゴリを管理するリストを用いて，複数のクラスをカテゴリを同時に管理する．
    ・Trainerからappendされる際は基本的にそのまま足していく．
    ・容量を大幅に超過するか，明示的に呼び出された場合に経験を捨てる．
    ・外部からバッチを要求される場合のAPIは若干変える．

    Prioritized Experience Replay：優先度付経験再生


    '''
    def __init__(self, capacity=10**4, transfer_threshold=10, capacity_threshold=2*10**4,
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
        if len(self) == 0:
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
        self.model = agent.model
        self.exp_buffer = exp_buffer if exp_buffer is not None else StratifiedExperienceBuffer()

        # 学習用に出力だけ結合したモデルを作成する．
        output_concat = layers.concatenate([layer for layer in self.model.outputs], axis=1)
        self.model_train = models.Model(self.model.inputs, output_concat)

        # ログの初期化
        self.loss_history = {'all':[], 'actor':[], 'critic':[], 'entropy':[]}
        self.return_history = {'all':[]}
        self.loss_color = {'all':'r', 'actor':'k', 'critic':'g', 'entropy':'b'}  # for graph
        self.return_color = {'all':'b'}  # for graph

        self.log_exp = {'state_current': [],
                        'action': [],
                        'reward': [],
                        'state_next': [],
                        'done': [],
                        'info': [],
                        'mu': [],
                        'sd': [],
                        'value': []}
        self.log_eva = {'state_current': [],
                        'action': [],
                        'reward': [],
                        'state_next': [],
                        'done': [],
                        'info': [],
                        'mu': [],
                        'sd': [],
                        'value': []}


    def train(self, n_steps=10000, training_interval=10, n_batches=10, batch_size=256,
              buf_alpha=0.5, buf_beta=0.5, buf_mode='rank',
              gamma=1.0,
              optimizer=optimizers.Adam(1e-2),
              mult_actor=1.0, mult_critic=1.0, mult_entropy=0.1, mult_beta=1.0,
              sigma_start=1.0, sigma_end=0.1, sigma_interval=10000,
              warmup_step=500,
              category_weights=None, default_weight=1.0,
              verbose=True, verbose_interval=100, evaluate_interval=100,
              get_log=False):

        self.mult_actor = mult_actor
        self.mult_critic = mult_critic
        self.mult_entropy = mult_entropy
        self.mult_beta = mult_beta

        # compile model for training.
        self.model_train.compile(optimizer=optimizer, loss=self.loss_all,
                                 metrics=[self.loss_actor, self.loss_critic, self.loss_entropy])

        # setting exp_buffer
        self.exp_buffer.reset()
        self.exp_buffer.gamma = gamma
        self.exp_buffer.n_advantages = 10  # virtually, episode end.
        self.exp_buffer.category_weights = category_weights
        self.exp_buffer.default_weight = default_weight

        # reset log
        self.reset_history()
        self.reset_log()

        loss = None
        for t in tqdm(range(n_steps)):
            # interaction with environment
            state_current = self.env.state()
            sigma = np.clip(sigma_start + (sigma_end - sigma_start) / sigma_interval * t,
                            np.minimum(sigma_start, sigma_end), np.maximum(sigma_start, sigma_end)).ravel()[0]
            if get_log:
                action, mu, sd, value = self.agent.get_action(state_current, greedy=False, get_log=get_log)
            else:
                action = self.agent.get_action(state_current, greedy=False, get_log=get_log)
            state_next, reward, info, done = self.env.step(action)
            self.exp_buffer.append([state_current, action, reward, state_next, done, info])

            # update model
            if (t+1) % training_interval == 0 and warmup_step < t:
                loss = self._train_on_batch(n_batches, batch_size,
                                            buf_alpha, buf_beta, buf_mode)
                if loss is not None:
                    # update loss log
                    self.loss_history['all'].append(loss[0])
                    self.loss_history['actor'].append(loss[1])
                    self.loss_history['critic'].append(loss[2])
                    self.loss_history['entropy'].append(loss[3])

            # log
            if get_log:
                self.log_exp['state_current'].append(state_current)
                self.log_exp['action'].append(action)
                self.log_exp['reward'].append(reward)
                self.log_exp['state_next'].append(state_next)
                self.log_exp['done'].append(done)
                self.log_exp['info'].append(info)
                self.log_exp['mu'].append(mu)
                self.log_exp['sd'].append(sd)
                self.log_exp['value'].append(value)

            # verbose
            if verbose and (t+1) % verbose_interval == 0 and loss is not None:
                print('step = {}, loss_all = {:.5f}, loss_actor = {:.5f}, loss_critic = {:.5f}, loss_entropy = {:.5f}'
                      .format(t, loss[0], loss[1], loss[2], loss[3]))

            # evaluate
            if (t+1) % evaluate_interval == 0:
                self.return_history['all'].append(self.evaluate(show_log=verbose, gamma=gamma, get_log=get_log))
                self.exp_buffer.reset_int_buffer()

        # lossをNDArrayに変換
        self.loss_history = {key: np.array(self.loss_history[key]) for key in self.loss_history.keys()}
        self.return_history = {key: np.array(self.return_history[key]) for key in self.return_history.keys()}

        # log統合
        if get_log:
            for key in ['state_current', 'state_next']:
                pass
                self.log_exp[key] = [np.concatenate(S, axis=0) for S in zip(*self.log_exp[key])]
                self.log_eva[key] = [np.concatenate(S, axis=0) for S in zip(*self.log_eva[key])]

            for key in ['action', 'reward', 'done', 'info', 'mu', 'sd', 'value']:
                self.log_exp[key] = np.concatenate(self.log_exp[key], axis=0)
                self.log_eva[key] = np.concatenate(self.log_eva[key], axis=0)


    def reset_history(self):
        self.loss_history = {key: [] for key in self.loss_history.keys()}
        self.return_history = {key: [] for key in self.return_history.keys()}


    def reset_log(self):
        self.log_exp = {key: [] for key in self.log_exp.keys()}
        self.log_eva = {key: [] for key in self.log_eva.keys()}


    def _train_on_batch(self, n_batches, batch_size,
                        buf_alpha, buf_beta, buf_mode):
        if len(self.exp_buffer) == 0:
            return None

        # model update
        losses = []
        for i in range(n_batches):
            S_c, A_c, R, S_n, done, pvf, weight, idx = \
                self.exp_buffer.get_batch(batch_size, buf_alpha, buf_beta, buf_mode)
            y_true = np.concatenate([A_c, R], axis=1)
            losses.append(self.model_train.train_on_batch([*S_c], y_true))  # lossは分けて出てくるので注意．

            # push back to PER
            priority = R - self.model.predict_on_batch([*S_c])[1]
            self.exp_buffer.update_priority(idx, priority)

        # consolidate loss and return
        return [np.mean(np.array(loss)) for loss in zip(*losses)]
        # dimension of losses is , n_batchs, loss_type, n_agents


    def loss_all(self, y_true, y_pred):
        '''
        AWR has 3 loss functions (originally 2, but this class was made from A2C).
        1.Policy loss : log policy(A|S) exp(Advantage/beta) with keep Advantage constant.
        2.Value loss : (MC value - model value)^2
        3.Entropy loss : dummy
        :param y_true:
        :param y_pred:
        :return:
        '''
        return self.loss_actor(y_true, y_pred) \
             + self.loss_critic(y_true, y_pred) \
             + self.loss_entropy(y_true, y_pred)


    def loss_actor(self, y_true, y_pred):
        # log Policy(A) * advantage. this aims increase the ratio of prefered action.
        action = y_true[:, :-1]  # actual action on-hot
        action_prob = y_pred[:, :-1]  # action prob
        action_prob_mu = action_prob[:, :K.int_shape(action_prob)[-1]//2]
        action_prob_sd = action_prob[:, K.int_shape(action_prob)[-1]//2:]
        v_target = y_true[:, -1:]  # n-step V at action a
        v_pred = y_pred[:, -1:]  # V pred by model

        p1 = ((action - action_prob_mu) ** 2) / (2 * (action_prob_sd ** 2) + K.epsilon())
        p2 = K.log(K.sqrt(2 * math.pi * ((action_prob_sd ** 2) + K.epsilon())))
        A = K.exp((v_target - v_pred) / self.mult_beta)

        return self.mult_actor * (p1 + p2) * K.stop_gradient(A)


    def loss_critic(self, y_true, y_pred):
        # value iteration part
        v_target = y_true[:, -1:]  # MC V
        v_pred = y_pred[:, -1:]  # V pred by model_value
        return self.mult_critic * K.sum(K.square(v_target-v_pred), axis=-1, keepdims=True)


    def loss_entropy(self, y_true, y_pred):
        # tuning exploration
        action_prob = y_pred[:, :-1]
        action_prob_sd = action_prob[:, K.int_shape(action_prob)[-1]//2:]  # N * n_actions
        return self.mult_entropy * K.sum((-(K.log(2 * math.pi * (action_prob_sd ** 2))+1))/2, axis=-1, keepdims=True)


    def evaluate(self, show_log=False, time_horizon=1000, n_showing_agents=10, gamma=1.0,
                 get_log=False):
        self.env.reset()
        rewards = []
        done_mask = 1  # ループ中にNDArrayでブロードキャストされるので，これで問題無い．
        for t in range(time_horizon):
            state_current = self.env.state()
            if get_log:
                action, mu, sd, value = self.agent.get_action(state_current, greedy=True, get_log=True)
            else:
                action = self.agent.get_action(state_current, greedy=True, get_log=False)
            state_next, reward, info, done = self.env.step(action)
            rewards.append(done_mask * reward)  # apply done_mask for cut off already done agents
            done_mask = done_mask * (1 - np.array(done, dtype=int)) * gamma  # update done_mask

            # log
            if get_log:
                self.log_eva['state_current'].append(state_current)
                self.log_eva['action'].append(action)
                self.log_eva['reward'].append(reward)
                self.log_eva['state_next'].append(state_next)
                self.log_eva['done'].append(done)
                self.log_eva['info'].append(info)
                self.log_eva['mu'].append(mu)
                self.log_eva['sd'].append(sd)
                self.log_eva['value'].append(value)

            if show_log:
                print('action = ', action.ravel()[:n_showing_agents])

            if done_mask.sum() == 0:  # end if all agents are done
                break

        rewards = np.array(rewards)[:, :, 0]  # T, n_agents, 1

        if show_log:
            print('return', rewards.sum(axis=0)[:n_showing_agents])

        return np.mean(rewards.sum(axis=0))  # n_agents


########################################################################################################################
class Tester(object):
    def __init__(self, env, model, action_min=-np.inf, action_max=np.inf, name='tester_AWR'):
        self.env = env
        self.model = model
        self.agent = Agent(model, action_min, action_max)
        self.trainer = Trainer(env, self.agent)
        self.name = name


    def test(self, n_trials=5, n_steps=10000, training_interval=10, n_batches=10, batch_size=256,
             buf_alpha=0.5, buf_beta=0.5, buf_mode='rank',
             gamma=1.0,
             optimizer=optimizers.Adam(1e-2),
             mult_actor=1.0, mult_critic=1.0, mult_entropy=0.1, mult_beta=1.0,
             sigma_start=1.0, sigma_end=0.1, sigma_interval=10000,
             warmup_step=500,
             category_weights=None, default_weight=1.0,
             verbose=False, verbose_interval=100, evaluate_interval=100,
             get_log=False, save_objects=False):

        print('start testing... : ', datetime.now())
        self.loss_histories = []
        self.return_histories = []
        for trial in range(n_trials):
            print('start {}/{} th trial...'.format(trial, n_trials))
            self.trial(n_steps=n_steps, training_interval=training_interval,
                       n_batches=n_batches, batch_size=batch_size,
                       buf_alpha=buf_alpha, buf_beta=buf_beta, buf_mode=buf_mode,
                       gamma=gamma,
                       optimizer=optimizer,
                       mult_actor=mult_actor, mult_critic=mult_critic, mult_entropy=mult_entropy, mult_beta=mult_beta,
                       sigma_start=sigma_start, sigma_end=sigma_end, sigma_interval=sigma_interval,
                       warmup_step=warmup_step,
                       category_weights=category_weights, default_weight=default_weight,
                       verbose=verbose, verbose_interval=verbose_interval, evaluate_interval=evaluate_interval,
                       get_log=get_log)

        print('end testing... : ', datetime.now())
        self.report(gamma=gamma)

        # saving
        with open(str(self.name) + '.pkl', 'wb') as f:
            if save_objects:
                pickle.dump([self.trainer , self.loss_histories, self.return_histories], f)
            else:
                pickle.dump([self.loss_histories, self.return_histories], f)


    def trial(self, n_steps, training_interval, n_batches, batch_size,
             buf_alpha, buf_beta, buf_mode,
             gamma,
             optimizer,
             mult_actor, mult_critic, mult_entropy, mult_beta,
             sigma_start, sigma_end, sigma_interval,
             warmup_step,
             category_weights, default_weight,
             verbose, verbose_interval, evaluate_interval,
             get_log):
        # reset models & optimizer
        self.model = models.Model.from_config(self.model.get_config())
        self.agent = Agent(self.model, self.agent.action_min, self.agent.action_max)
        self.trainer = Trainer(self.env, self.agent)
        optimizer = optimizer.from_config(optimizer.get_config())

        # train
        self.trainer.train(n_steps=n_steps, training_interval=training_interval,
                           n_batches=n_batches, batch_size=batch_size,
                           buf_alpha=buf_alpha, buf_beta=buf_beta, buf_mode=buf_mode,
                           gamma=gamma,
                           optimizer=optimizer,
                           mult_actor=mult_actor, mult_critic=mult_critic, mult_entropy=mult_entropy, mult_beta=mult_beta,
                           sigma_start=sigma_start, sigma_end=sigma_end, sigma_interval=sigma_interval,
                           warmup_step=warmup_step,
                           category_weights=category_weights, default_weight=default_weight,
                           verbose=verbose, verbose_interval=verbose_interval, evaluate_interval=evaluate_interval,
                           get_log=get_log)

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
        self.trainer.evaluate(show_log=True, n_showing_agents=10, gamma=gamma, get_log=False)


    def rep_param(self, figsize=(15, 5), dpi=100):
        print('param distribution in model')
        p_layers = [layer for layer in self.model.layers if len(layer.get_weights()) != 0]

        fig = plt.figure(figsize=figsize, dpi=dpi)
        for i, layer in enumerate(p_layers):
            ax1 = fig.add_subplot(2, len(p_layers), 1 + i)
            ax1.hist(layer.get_weights()[0].ravel(), bins=20)
            ax1.set_title(layer.name + ' : W')

            ax2 = fig.add_subplot(2, len(p_layers), 1 + len(p_layers) + i)
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


    def viz_P_history(self, trainer, data_span_exp=10, data_span_eva=10,
                      figsize=(15, 5), dpi=150, cmap='jet', fontsize=16):
        plt.figure(figsize=figsize, dpi=dpi)
        sns.heatmap(trainer.log_exp['mu'][::data_span_exp].T, cmap=cmap)
        plt.title('evolution of Policy in exploration', size=fontsize)
        plt.xlabel('data step', size=fontsize)
        plt.ylabel('action', size=fontsize)
        plt.show()

        plt.figure(figsize=figsize, dpi=dpi)
        sns.heatmap(trainer.log_eva['mu'][::data_span_eva].T, cmap=cmap)
        plt.title('evolution of Policy in evaluation', size=fontsize)
        plt.xlabel('data step', size=fontsize)
        plt.ylabel('action', size=fontsize)
        plt.show()


########################################################################################################################
if __name__ == '__main__':
    n_agents = 2
    state_dims = [1, 3]
    n_outputs = 1

    env = Environment(n_agents=n_agents, state_dims=state_dims, n_outputs=n_outputs)

    model = gen_model(input_shapes=[[d,] for d in state_dims],
                      n_outputs=n_outputs,
                      hidden_dims=[2, 4, 6])
    agent = Agent(model)
    agent.get_action(env.state())
    #trainer = Trainer(env, agent)
    #trainer.train(verbose=False, gamma=0.95)

    tester = Tester(env, model)
    tester.test()


    print('END')