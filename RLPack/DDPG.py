import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras import layers
from keras import regularizers
from keras import models
from keras import optimizers
import keras.backend as K

import scipy.stats  # used in exp_buffer for PER.

from tqdm import tqdm
import warnings
# warnings.filterwarnings('ignore')  # set for ignore keras trainable setting warnings.

from datetime import datetime
import pickle



'''
DDPG (Deep Deterministic Policy Gradient)
https://spinningup.openai.com/en/latest/algorithms/ddpg.html

DDPG is like continuous version of DQN. But theoritical basis is coming from PG (or deterministic policy gradient)



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
            #state += (action==1) * np.random.uniform(size=state.shape)  # *= で入れないと無効．
            #state += (action==1) * np.random.randint(0, 2+1, size=(state.shape))
        #reward = np.random.uniform(-1, 1, self.done.shape)
            pass
        reward = np.random.randint(-1, 1+1, self.done.shape)
        self.done = np.random.binomial(1, 0.2, self.done.shape)

        return [self.state(), np.copy(reward), None, np.copy(self.done)]


    def state(self):
        # N*S_0, N*S_1,...
        return [np.copy(state) for state in self.states]


    def reset(self):
        self.states = []
        for state_dim in self.state_dims:
            self.states.append(np.arange(state_dim) * np.ones(self.n_agents).reshape(-1, 1))  # n_agents, state_dim
        self.done = np.ones(shape=(self.n_agents, 1), dtype=bool)  # n_agents, 1


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
        self.model_value = model_value  # valueは本来は不要だが，APIを揃えたいのでここで取る．
        self.val_min = val_min ; self.val_max = val_max
        self.count = 0


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
class ExperienceBuffer(object):
    # dimS*N*S, N*A, N*R, dimS*N*S', N*doneで管理する．
    def __init__(self, max_size=10**4):
        self.max_size = max_size
        self.reset()


    def reset(self):
        self.state_current = None
        self.action = None
        self.reward = None
        self.state_next = None
        self.done = None

        self.idx = 0 # index on buffer
        self.need_shuffle = True


    def get_batch(self, batch_size=512):
        if len(self) <= self.idx or self.need_shuffle==True:  # idxが終端に達しているか新要素が追加されている．
            self.shuffle()
            self.idx = 0
        batch_number = self.number[self.idx: self.idx + batch_size]
        state_current = [state[batch_number] for state in self.state_current]
        state_next = [state[batch_number] for state in self.state_next]
        action = self.action[batch_number]
        reward = self.reward[batch_number]
        done = self.done[batch_number]

        self.idx += batch_size

        return state_current, action, reward, state_next, done


    def __len__(self):
        if self.state_current is None:
            return 0
        else:
            return len(self.state_current[0])


    def shuffle(self):
        if len(self)==0:
            return None
        self.number = np.random.permutation(len(self))
        self.need_shuffle = False


    def append(self, experience):
        state_current, action, reward, state_next, done = experience
        self.need_shuffle = True
        if len(self)==0:
            self.state_current = [state_ex for state_ex in state_current]
            self.state_next = [state_ex for state_ex in state_next]
            self.action = action
            self.reward = reward
            self.done = done
            return None

        self.state_current = [np.concatenate([state_in, state_ex], axis=0)[-self.max_size: ] \
                              for state_in, state_ex in zip(self.state_current, state_current)]
        self.state_next = [np.concatenate([state_in, state_ex], axis=0)[-self.max_size: ] \
                              for state_in, state_ex in zip(self.state_next, state_next)]
        self.action = np.concatenate([self.action, action], axis=0)[-self.max_size: ]
        self.reward = np.concatenate([self.reward, reward], axis=0)[-self.max_size: ]
        self.done = np.concatenate([self.done, done], axis=0)[-self.max_size: ]


########################################################################################################################

class Trainer(object):
    def __init__(self, env, agent, exp_buffer=None):
        self.env = env
        self.agent = agent
        self.model_policy = agent.model_policy
        self.model_value = agent.model_value
        self.exp_buffer = exp_buffer if exp_buffer is not None else ExperienceBuffer()

        self.model_policy_target = models.Model.from_config(self.model_policy.get_config())  # 同一フォームのモデルを生成
        self.model_value_target = models.Model.from_config(self.model_value.get_config())  # 同一フォームのモデルを生成
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


    def train(self, gamma,
              n_steps, training_interval, n_batches, batch_size,
              target_update_interval_policy, target_update_interval_value,
              tau_policy, tau_value,
              optimizer_pv, optimizer_value,
              sigma_start, sigma_end, sigma_interval,
              boltzmann, tau_start, tau_end, tau_interval, n_samples,
              verbose=True, verbose_interval=100, evaluate_interval=1000,
              warmup_steps=500,
              get_log=False):

        # model形成
        self.model_value.compile(optimizer=optimizer_value, loss='mse')
        self.model_value.trainable = False
        model_pv_output = self.model_value([*self.model_value.inputs][:-1]
                                          +[self.model_policy([*self.model_value.inputs][:-1]), ])
        self.model_pv = models.Model([*self.model_value.inputs][:-1], model_pv_output)
        self.model_pv.compile(optimizer=optimizer_pv, loss=self.loss_pv)

        self.reset_history()
        self.reset_log()
        loss = None

        for t in tqdm(range(n_steps)):
            state_current = self.env.state()
            sigma = max(sigma_end, sigma_start + (sigma_end - sigma_start) / sigma_interval * t)
            tau = max(tau_end, tau_start + (tau_end - tau_start) / tau_interval * t)
            if get_log:
                action, Q, action_sample = self.agent.get_action(state_current, sigma, False, boltzmann, n_samples, tau, get_log=get_log)
            else:
                action = self.agent.get_action(state_current, sigma, False, boltzmann, n_samples, tau, get_log=get_log)
            state_next, reward, info, done = self.env.step(action)
            self.exp_buffer.append([state_current, action, reward, state_next, done])

            # update
            if (t+1) % training_interval == 0 and warmup_steps < t:
                loss = self._train_on_batch(n_batches, batch_size, gamma)
                self.loss_history['all'].append(loss[0] + loss[1])
                self.loss_history['pv'].append(loss[0])
                self.loss_history['value'].append(loss[1])

            # model_target_policy更新
            if (t+1) % target_update_interval_policy == 0:
                weights_base = self.model_policy.get_weights()
                weights_target = self.model_policy_target.get_weights()
                new_weights = [(weight_base * tau_policy + weight_target * (1 - tau_policy)) \
                               for weight_base, weight_target in zip(weights_base, weights_target)]
                self.model_policy_target.set_weights(new_weights)

            # model_target_value更新
            if (t+1) % target_update_interval_value == 0:
                weights_base = self.model_value.get_weights()
                weights_target = self.model_value_target.get_weights()
                new_weights = [(weight_base * tau_value + weight_target * (1 - tau_value)) \
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
                print('step = {}, loss_all = {:.5f}, loss_pv = {:.5f}, loss_value = {:.5f}' \
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


    def _train_on_batch(self, n_batches, batch_size, gamma):
        if len(self.exp_buffer) == 0:
            return None

        losses_pv = []
        losses_value = []
        for i in range(n_batches):
            # replay_bufferからサンプリング
            S_c, A_c, R, S_n, done = self.exp_buffer.get_batch(batch_size)

            # target生成
            A_n = self.model_policy_target.predict([*S_n])
            Q_pred_next = self.model_value_target.predict([*S_n] + [A_n, ])
            Q_pred_next[done==True] = 0.0
            target = R.ravel() + gamma * Q_pred_next.ravel()

            # model_policy更新をmodel_pvを通して行う
            loss_pv = self.model_pv.train_on_batch([*S_c], target)  # targetはダミーとして入れておく

            # model_value更新
            with warnings.catch_warnings():  # Kerasの仕様でwarningsが出るため，一時的に解除する．
                warnings.simplefilter('ignore')
                loss_value = self.model_value.train_on_batch([*S_c] + [A_c, ], target)

            losses_pv.append(loss_pv)
            losses_value.append(loss_value)

        losses_pv = np.array(losses_pv).mean()
        losses_value = np.array(losses_value).mean()
        return losses_pv, losses_value


    def loss_pv(self, y_true, y_pred):
        return -y_pred


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
    def __init__(self, env, model_policy, model_value, val_min, val_max, name='tester_DDPG'):
        self.env = env
        self.model_policy = model_policy
        self.model_value = model_value
        self.agent = Agent(model_policy, model_value, val_min, val_max)
        self.trainer = Trainer(env, self.agent)

        self.name = name


    def test(self, n_trials=5,
             gamma=1.0,
             n_steps=10000, training_interval=10, n_batches=10, batch_size=64,
             target_update_interval_policy=20, target_update_interval_value=20,
             tau_policy=0.001, tau_value=0.001,
             optimizer_pv=optimizers.RMSprop(), optimizer_value=optimizers.RMSprop(),
             sigma_start=1.0, sigma_end=1.0, sigma_interval=10000,
             boltzmann=False, tau_start=0.1, tau_end=1.0, tau_interval=10000, n_samples=128,
             verbose=False, verbose_interval=100, evaluate_interval=1000,
             wampup_steps=500,
             get_log=False, save_objects=False):

        print('start testing... : ', datetime.now())
        self.loss_histories = []
        self.return_histories = []
        for trial in range(n_trials):
            print('start {}/{} th trial...'.format(trial, n_trials))
            self.trial(gamma,
                       n_steps, training_interval, n_batches, batch_size,
                       target_update_interval_policy, target_update_interval_value,
                       tau_policy, tau_value,
                       optimizer_pv, optimizer_value,
                       sigma_start, sigma_end, sigma_interval,
                       boltzmann, tau_start, tau_end, tau_interval, n_samples,
                       verbose, verbose_interval, evaluate_interval,
                       wampup_steps,
                       get_log)

        print('end testing... : ', datetime.now())
        self.report(gamma=gamma)

        # 保存処理
        with open(str(self.name) + '.pkl', 'wb') as f:
            if save_objects:
                pickle.dump([self.trainer , self.loss_histories, self.return_histories], f)
            else:
                pickle.dump([self.loss_histories, self.return_histories], f)


    def trial(self, gamma,
              n_steps, training_interval, n_batches, batch_size,
              target_update_interval_policy, target_update_interval_value,
              tau_policy, tau_value,
              optimizer_pv, optimizer_value,
              sigma_start, sigma_end, sigma_interval,
              boltzmann, tau_start, tau_end, tau_interval, n_samples,
              verbose, verbose_interval, evaluate_interval,
              warmup_steps,
              get_log):

        # model等のリセット
        self.model_policy = models.Model.from_config(self.model_policy.get_config())
        self.model_value = models.Model.from_config(self.model_value.get_config())
        self.agent = Agent(self.model_policy, self.model_value, self.agent.val_min, self.agent.val_max)
        self.trainer = Trainer(self.env, self.agent)

        # optimizerのリセット
        optimizer_pv = optimizer_pv.from_config(optimizer_pv.get_config())
        optimizer_value = optimizer_value.from_config(optimizer_value.get_config())

        # 学習
        self.trainer.train(gamma=gamma,
                           n_steps=n_steps, training_interval=training_interval, n_batches=n_batches, batch_size=batch_size,
                           target_update_interval_policy=target_update_interval_policy, target_update_interval_value=target_update_interval_value,
                           tau_policy=tau_policy, tau_value=tau_value,
                           optimizer_pv=optimizer_pv, optimizer_value=optimizer_value,
                           sigma_start=sigma_start, sigma_end=sigma_end, sigma_interval=sigma_interval,
                           boltzmann=boltzmann, tau_start=tau_start, tau_end=tau_end, tau_interval=tau_interval, n_samples=n_samples,
                           verbose=verbose, verbose_interval=verbose_interval, evaluate_interval=evaluate_interval,
                           warmup_steps=warmup_steps,
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
    n_actions = 3
    action_min = -0.1
    action_max = 1.1

    env = Environment(n_agents=n_agents, state_dims=state_dims, n_actions=n_actions)

    model_policy = gen_model_policy(state_shapes=[[d, ] for d in state_dims],
                                    n_actions=n_actions,
                                    hidden_dims=[2, 4],
                                    reg_l1=0.01, reg_l2=0.02,
                                    action_min=action_min, action_max=action_max)

    model_value = gen_model_value(state_shapes=[[d, ] for d in state_dims],
                                  n_actions=n_actions,
                                  hidden_action_dims=[10, 5], hidden_state_dims=[3, 3], hidden_dims=[3, 3],
                                  reg_l1=0.01, reg_l2=0.01)

    #model_value.summary()
    #model_value.compile(optimizer='sgd', loss='mse')
    #model_value.trainable = False

    #inp_0 = layers.Input([10])
    #inp_1 = layers.Input([20])
    #inp_2 = layers.Input([1])
    #z = model_policy([*model_value.inputs][:-1])

    #q_end = model_value([inp_0, inp_1, model_policy([inp_0, inp_1])])
    #print('XXXX', [*model_value.inputs][:-1])
    #print('YYYY', model_policy([*model_value.inputs][:-1]))
    #print('ZZZZ', [*model_value.inputs][:-1] + [model_policy([*model_value.inputs][:-1]), ])
    #q_end = model_value([*model_value.inputs][:-1] + [model_policy([*model_value.inputs][:-1]), ])
    agent = Agent(model_policy, model_value, val_min=0.0, val_max=10.0)
    trainer = Trainer(env, agent)
    tester = Tester(env, model_policy, model_value, val_min=0.0, val_max=10.0)
    tester.test(n_trials=1, boltzmann=True)





    print('END')
