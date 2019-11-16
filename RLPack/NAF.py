import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras import layers
from keras import models
from keras import regularizers
from keras import optimizers
import keras.backend as K  # used in model.

from datetime import datetime
import time
from tqdm import tqdm
import pickle


'''
NAF


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
def gen_model(input_shapes=((1, ), (10, ), (10, )), n_actions=10,
              hidden_dims=(256, 128, 64), reg_l1=0.0, reg_l2=0.0,
              input_reg=False, input_min=-10, input_max=10,
              action_reg=False, action_min=-10, action_max=10,
              value_reg=False, value_min=0, value_max=10):
    '''
    NAF has 3 outputs...V, L, mu.
    V : Value function of state s.
    L : lower triangle matrix for creating
    mu : policy
    therefore, this part is the most big difference compared to DQN.
    :param input_shapes:
    :param n_actions:
    :param hidden_dims:
    :param reg_l1:
    :param reg_l2:
    :param input_reg:
    :param input_min:
    :param input_max:
    :param action_reg:
    :param action_min:
    :param action_max:
    :param value_reg:
    :param value_min:
    :param value_max:
    :return:
    '''

    # input
    input_ts = [layers.Input(input_shape) for input_shape in input_shapes]
    if 2 <= len(input_ts):
        input_concat = layers.concatenate(input_ts, axis=-1)
    else:
        input_concat = layers.Lambda(lambda x: x)(input_ts[0])  # 恒等レイヤー

    if input_reg:
        input_concat = layers.Lambda(lambda x: (x - input_min) / (input_max - input_min))(input_concat)

    # hidden
    for i, hidden_dim in enumerate(hidden_dims):
        if i == 0:
            x = layers.LeakyReLU(alpha=0.1)(layers.Dense(hidden_dim, kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(input_concat))
        else:
            x = layers.LeakyReLU(alpha=0.1)(layers.Dense(hidden_dim, kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x))

    # NAF specific
    input_action = layers.Input([n_actions, ])  # action input
    if action_reg:
        a = layers.Lambda(lambda z: (z - action_min) / (action_max - action_min))(input_action)
    else:
        a = layers.Lambda(lambda z: z)(input_action)

    mu = layers.LeakyReLU(alpha=0.1)(layers.Dense(n_actions)(x))  # action output

    L_diagonal = layers.LeakyReLU(alpha=0.1)(layers.Dense(n_actions)(x))
    L_offdiag = layers.LeakyReLU(alpha=0.1)(layers.Dense((n_actions-1) * n_actions // 2)(x))
    L_diagonal = layers.Lambda(lambda z: K.exp(z))(L_diagonal)

    L = []
    for d in range(n_actions):
        if d == 0:
            L.append(layers.Lambda(lambda z: z[:, d:d + 1])(L_diagonal))
            L.append(layers.Lambda(lambda z: K.zeros(shape=(K.shape(z)[0], n_actions - (d + 1))))(L_diagonal))
        else:
            L.append(layers.Lambda(lambda z: z[:, d * (d-1) // 2: d * (d-1) // 2 + d])(L_offdiag))  # 0, 1 : 1, 3 : 3,6
            L.append(layers.Lambda(lambda z: z[:, d:d + 1])(L_diagonal))
            if d != n_actions - 1:
                L.append(layers.Lambda(lambda z: K.zeros(shape=(K.shape(z)[0], n_actions - (d + 1))))(L_diagonal))

    L = layers.concatenate(L)
    L = layers.Lambda(lambda z: K.reshape(z, (K.shape(z)[0], n_actions, n_actions)))(L)
    P = layers.Lambda(lambda z: K.batch_dot(layers.Permute((2, 1))(z), z))(L)

    A = layers.Lambda(lambda z: -(1/2) * K.batch_dot(K.batch_dot(K.expand_dims(a - mu, 1), z), a - mu))(P)  # 厳密にはadvantageではないので，標準化してはいけない．
    V = layers.LeakyReLU(alpha=0.1)(layers.Dense(1, )(x))
    Q = layers.Add()([V, A])

    # adjust for output
    if action_reg:
        mu = layers.Activation('sigmoid')(mu)
        mu = layers.Lambda(lambda z: action_min + (action_max - action_min) * z)(mu)
    if value_reg:
        V = layers.Lambda(lambda z: value_min + (value_max - value_min) * z)(V)
        Q = layers.Lambda(lambda z: value_min + (value_max - value_min) * z)(Q)

    return models.Model(input_ts + [input_action, ], [mu, V, Q])



########################################################################################################################
class Agent(object):
    def __init__(self, model, val_min=0, val_max=1):
        self.model = model
        self.n_actions = model.inputs[-1].shape[1]

        self.val_min = val_min
        self.val_max = val_max


    def get_action(self, state, sigma=1.0, adaptive=False, greedy=False, get_log=False):
        dummy_action = np.zeros(shape=(len(state[0]), self.n_actions))
        mu, _, _ = self.model.predict_on_batch([*state] + [dummy_action,])
        if greedy:
            action = np.clip(mu, self.val_min, self.val_max)
        else:
            Z = np.random.randn(*mu.shape)
            action = np.clip(mu + np.array(sigma) * Z, self.val_min, self.val_max)

        return action


########################################################################################################################
class ExperienceBuffer(object):
    # [n*S]*dim_S, n*A, n*R, [n*S']*dim_S', n*done
    def __init__(self, max_size=10**4):
        self.max_size = max_size
        self.reset()


    def reset(self):
        self.state_current = None
        self.action = None
        self.reward = None
        self.state_next = None
        self.done = None

        self.number = None
        self.idx = 0  # index on buffer.
        self.need_shuffle = True


    def get_batch(self, batch_size=256):
        if len(self) <= self.idx or self.need_shuffle:
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
        if len(self) == 0:
            return None
        self.number = np.random.permutation(len(self))
        self.need_shuffle = False


    def append(self, experience):
        # experience = [n*S]*dim_S, n*A, n*R, [n*S']*dim_S', n*done
        state_current, action, reward, state_next, done = experience
        self.need_shuffle = True
        if len(self) == 0:
            self.state_current = [state_ex for state_ex in state_current]
            self.state_next = [state_ex for state_ex in state_next]
            self.action = action
            self.reward = reward
            self.done = done
            return None

        self.state_current = [np.concatenate([state_in, state_ex], axis=0)[-self.max_size:]
                              for state_in, state_ex in zip(self.state_current, state_current)]
        self.state_next = [np.concatenate([state_in, state_ex], axis=0)[-self.max_size:]
                              for state_in, state_ex in zip(self.state_next, state_next)]
        self.action = np.concatenate([self.action, action], axis=0)[-self.max_size:]
        self.reward = np.concatenate([self.reward, reward], axis=0)[-self.max_size:]
        self.done = np.concatenate([self.done, done], axis=0)[-self.max_size:]


########################################################################################################################
class Trainer(object):
    def __init__(self, env, agent, exp_buffer=None, model_gen_func=None):
        self.env = env
        self.agent = agent
        self.model = self.agent.model
        self.model_gen_func = model_gen_func
        self.model_target = self.model_gen_func()  # keras側の問題でmodelのコピーができないのでとりあえず封印．
        # self.model_target = models.Model.from_config(self.model.get_config())  # 同一フォームのモデルを生成

        self.exp_buffer = exp_buffer if exp_buffer is not None else ExperienceBuffer()

        self.loss_history = {'all':[]}
        self.return_history = {'all':[]}
        self.loss_color = {'all':'r'}  # for graph
        self.return_color = {'all':'b'}  # for graph

        self.log_exp = {'state_current':[],
                        'action':[],
                        'reward':[],
                        'state_next':[],
                        'done':[],
                        'info':[]}
        self.log_eva = {'state_current': [],
                        'action': [],
                        'reward': [],
                        'state_next': [],
                        'done': [],
                        'info': []}


    def train(self, n_steps=10000, training_interval=10, n_batches=10, batch_size=256,
              alpha=1.0, target_update_interval=20,
              gamma=1.0,
              optimizer=optimizers.Adam(1e-2),
              sigma_start=1.0, sigma_end=0.1, sigma_interval=10000,
              verbose=True, verbose_interval=10, evaluate_interval=100,
              warmup_steps=500,
              get_log=False):

        self.model.compile(optimizer=optimizer, loss=self._loss_value)

        self.exp_buffer.reset()
        self.reset_history()
        self.reset_log()
        loss = None

        for t in tqdm(range(n_steps)):
            sigma = np.clip(sigma_start + (sigma_end - sigma_start) / sigma_interval * t,
                            np.minimum(sigma_start, sigma_end), np.maximum(sigma_start, sigma_end))
            s_current = self.env.state()
            action = self.agent.get_action(s_current, sigma, greedy=False, get_log=get_log)
            s_next, reward, info, done = self.env.step(action)
            self.exp_buffer.append([s_current, action, reward, s_next, done])

            # log
            if get_log:
                self.log_exp['state_current'].append(s_current)
                self.log_exp['action'].append(action)
                self.log_exp['reward'].append(reward)
                self.log_exp['state_next'].append(s_next)
                self.log_exp['done'].append(done)
                self.log_exp['info'].append(info)

            # update
            if (t+1) % training_interval == 0 and warmup_steps < t:
                loss = self._fit_on_batch(n_batches, batch_size, alpha, gamma)
                self.loss_history['all'].append(loss)

            if (t+1) % target_update_interval == 0:
                self.model_target.set_weights(self.model.get_weights())

            if verbose and (t+1) % verbose_interval == 0 and loss is not None:
                print('step = {}, loss = {:.5f}, sigma = {:.5f}'
                      .format(t, loss, sigma))

            if (t+1) % evaluate_interval == 0:
                self.return_history['all'].append(self.evaluate(show_log=verbose, gamma=gamma, get_log=get_log))

        # loss to NDArray
        self.loss_history = {key: np.array(self.loss_history[key]) for key in self.loss_history.keys()}
        self.return_history = {key: np.array(self.return_history[key]) for key in self.return_history.keys()}

        # log統合
        if get_log:
            for key in ['state_current', 'state_next']:
                pass
                self.log_exp[key] = [np.concatenate(S, axis=0) for S in zip(*self.log_exp[key])]
                self.log_eva[key] = [np.concatenate(S, axis=0) for S in zip(*self.log_eva[key])]

            for key in ['action', 'reward', 'done', 'info']:
                self.log_exp[key] = np.concatenate(self.log_exp[key], axis=0)
                self.log_eva[key] = np.concatenate(self.log_eva[key], axis=0)


    def reset_history(self):
        self.loss_history = {key:[] for key in self.loss_history.keys()}
        self.return_history = {key:[] for key in self.return_history.keys()}


    def reset_log(self):
        self.log_exp = {key:[] for key in self.log_exp.keys()}
        self.log_eva = {key:[] for key in self.log_eva.keys()}


    def _loss_value(self, y_true, y_pred):
        Q_true = y_true[2]
        Q_pred = y_pred[2]
        return (Q_true - Q_pred) ** 2


    def _fit_on_batch(self, n_batches, batch_size, alpha, gamma):
        if len(self.exp_buffer) == 0:
            return None

        losses = []
        for i in range(n_batches):
            S_c, A, R, S_n, done = self.exp_buffer.get_batch(batch_size)
            m, V_pred_curr, Q_pred_curr = self.model.predict_on_batch([*S_c, A])
            _, V_pred_next, Q_pred_next = self.model_target.predict_on_batch([*S_n, A])  # Q_pred_nextは不要なので，Aをそのまま入れている．
            V_pred_next[done==1] = 0.0
            target = Q_pred_curr + alpha * ((R + gamma * V_pred_next) - Q_pred_curr)
            target = [m, V_pred_curr, target]

            loss = self.model.train_on_batch([*S_c, A], target)
            losses.append(loss[0])  # 複数ヘッドのNNなので，lossはその分も分割されて出力されている．

        return np.array(losses).mean()


    def evaluate(self, show_log=False, time_horizon=1000, n_showing_agents=10, gamma=1.0,
                 get_log=False):
        self.env.reset()
        rewards = []
        done_mask = 1
        for t in range(time_horizon):
            state_current = self.env.state()
            action = self.agent.get_action(state_current, greedy=True, get_log=True)
            state_next, reward, info, done = self.env.step(action)
            rewards.append(done_mask * reward)
            done_mask = done_mask * (1 - np.array(done, dtype=int)) * gamma

            # log
            if get_log:
                self.log_eva['state_current'].append(state_current)
                self.log_eva['action'].append(action)
                self.log_eva['reward'].append(reward)
                self.log_eva['state_next'].append(state_next)
                self.log_eva['done'].append(done)
                self.log_eva['info'].append(info)

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
    def __init__(self, env, model, val_min=0, val_max=1, name='NAF', model_gen_func=None):
        self.env = env
        self.model = model
        self.agent = Agent(model, val_min, val_max)
        self.trainer = Trainer(self.env, self.agent, None, model_gen_func)
        self.model_gen_func = model_gen_func

        self.name = name


    def test(self, n_trials=5,
             n_steps=10000, training_interval=10, n_batches=10, batch_size=256,
             alpha=1.0, target_update_interval=20,
             gamma=1.0,
             optimizer=optimizers.Adam(1e-2),
             sigma_start=1.0, sigma_end=0.1, sigma_interval=10000,
             verbose=False, verbose_interval=10, evaluate_interval=100,
             warmup_steps=500,
             get_log=False, save_objects=False
             ):
        print('start testing... : ', datetime.now())
        self.loss_histories = []
        self.return_histories = []
        for trial in range(n_trials):
            print('start trial {}/{} trial...'.format(trial, n_trials))
            self.trial(n_steps=n_steps, training_interval=training_interval, n_batches=n_batches, batch_size=batch_size,
                       alpha=alpha, target_update_interval=target_update_interval,
                       gamma=gamma,
                       optimizer=optimizer,
                       sigma_start=sigma_start, sigma_end=sigma_end, sigma_interval=sigma_interval,
                       verbose=verbose, verbose_interval=verbose_interval, evaluate_interval=evaluate_interval,
                       warmup_steps=warmup_steps,
                       get_log=get_log)

        print('end testing... : ', datetime.now())
        self.report(gamma=gamma)

        # 保存処理
        with open(str(self.name) + '.pkl', 'wb') as f:
            if save_objects:
                pickle.dump([self.trainer , self.loss_histories, self.return_histories], f)
            else:
                pickle.dump([self.loss_histories, self.return_histories], f)


    def trial(self, n_steps, training_interval, n_batches, batch_size,
              alpha, target_update_interval,
              gamma,
              optimizer,
              sigma_start, sigma_end, sigma_interval,
              verbose, verbose_interval, evaluate_interval,
              warmup_steps,
              get_log):

        # reset model
        #self.model = models.Model.from_config(self.model.get_config())
        self.model = self.model_gen_func()
        self.agent = Agent(self.model, self.agent.val_min, self.agent.val_max)
        self.trainer = Trainer(self.env, self.agent, None, self.model_gen_func)  # exp_buffer will be reseted on trainer side.

        # reset optimizer
        optimizer = optimizer.from_config(optimizer.get_config())

        # 学習
        self.trainer.train(n_steps=n_steps, training_interval=training_interval, n_batches=n_batches, batch_size=batch_size,
                           alpha=alpha, target_update_interval=target_update_interval,
                           gamma=gamma,
                           optimizer=optimizer,
                           sigma_start=sigma_start, sigma_end=sigma_end, sigma_interval=sigma_interval,
                           verbose=verbose, verbose_interval=verbose_interval, evaluate_interval=evaluate_interval,
                           warmup_steps=warmup_steps,
                           get_log=get_log)

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
        print('loss and return history')

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


########################################################################################################################
if __name__ == '__main__':
    n_agents = 4
    state_dims = [1, 3]
    n_outputs = 10
    env = Environment(n_agents=n_agents, state_dims=state_dims, n_actions=n_outputs)
    #print(env.state())
    #print(env.step(0))
    #print(env.step(1))

    model_gen_func = lambda : gen_model(input_shapes=[[d, ] for d in state_dims], n_actions=n_outputs, hidden_dims=[2, 4, 6])
    model = model_gen_func()
    #agent = Agent(model)
    #action = agent.get_action(env.state())
    #print(action)
    #print(model.summary())

    #trainer = Trainer(env, agent)
    #trainer.train()
    tester = Tester(env, model, None, model_gen_func=model_gen_func)
    tester.test(n_trials=1, n_steps=1000)

    #tester = Tester(env, model)

