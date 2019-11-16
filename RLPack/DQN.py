import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras import layers
from keras import models
from keras import regularizers
from keras import optimizers
import keras.backend as K  # used in dueling network

from datetime import datetime
import time
from tqdm import tqdm
import pickle


'''
DQN
includes below features
 - double Q in Trainer.
 - dueling network in gen_model function.

'''


########################################################################################################################
class Environment(object):
    '''
    Dummy environment for testing.
    '''
    def __init__(self, n_agents=2, state_dims=[1, 3, 5], n_outputs=5):
        self.n_agents = n_agents
        self.state_dims = state_dims
        self.n_outputs = n_outputs
        self.reset()


    def step(self, action):
        for state in self.states:
            #state += (action==1) * np.random.uniform(size=state.shape)  # *= で入れないと無向．
            state += (action==1) * np.random.randint(0, 2+1, size=(state.shape))
        #reward = np.random.uniform(-1, 1, self.done.shape)
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
def gen_model(input_shapes=[[1], [10], [10]], n_outputs=10, \
              hidden_dims=[512, 256, 128, 64, 32], reg_l1=0.0, reg_l2=0.0, \
              duel=False, duel_value_dim=10, duel_advantage_dim=10,
              input_reg=False, input_min=-10, input_max=10, \
              output_reg=False, output_min=-10, output_max=10):
    '''
    :param input_shapes: list of input shapes.
    :param n_outputs: specify output action dimension.
    :param hidden_dims: list of newrons in each hidden layers.
    :return: Q value of each actions.
    neural net generator for function approximation of Q value.
    '''
    input_ts = [layers.Input(input_shape) for input_shape in input_shapes]
    if 2 <= len(input_ts):
        input_concat = layers.concatenate(input_ts, axis=-1)
    else:
        input_concat = layers.Lambda(lambda x: x)(input_ts[0])  # 恒等レイヤー．

    if input_reg:
        input_concat = layers.Lambda(lambda x: x/(input_max-input_min))(input_concat)

    for i, hidden_dim in enumerate(hidden_dims):
        if i == 0:
            x = layers.LeakyReLU(alpha=0.1)(layers.Dense(hidden_dim, kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(input_concat))
        else:
            x = layers.LeakyReLU(alpha=0.1)(layers.Dense(hidden_dim, kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x))

    if duel==True:
        value_path = layers.Dense(duel_value_dim, kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x)
        value_head = layers.Dense(1, kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(value_path)
        adv_path = layers.Dense(duel_advantage_dim, kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x)
        adv_head = layers.Dense(n_outputs, kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(adv_path)
        y = layers.concatenate([adv_head, value_head])
        Q_head = layers.Lambda(lambda a: K.expand_dims(a[:, -1], -1) + a[:, :-1] - K.stop_gradient(K.mean(a[:, :-1], keepdims=True)))(y)

    else:
        Q_head = layers.Dense(n_outputs, activation='linear', kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x)

    if output_reg:
        Q_head = layers.Lambda(lambda x: output_min + (output_max-output_min) * x)(Q_head)

    return models.Model(input_ts, [Q_head,])


########################################################################################################################
class Agent(object):
    def __init__(self, model):
        self.model = model  # keras NN object for approximating Q value.


    def get_action(self, state, epsilon=0.01, boltzmann=False, greedy=False, get_log=False):
        Q = self.model.predict([*state])  # n_agents, n_state_dims
        if greedy:
            epsilon = 0
            boltzmann = False

        if boltzmann==False:
            # epsilon greedy
            action = np.argmax(Q, axis=-1)  # n_agents
            explor = (np.random.uniform(size=Q.shape[0]) < epsilon)  # mask for exploring apply.
            if 0 < explor.sum():
                action[explor] = np.random.randint(0, Q.shape[-1], size=explor.sum())  # apply exploring
        else:
            # Boltzmann
            Q = Q - np.max(Q, axis=-1, keepdims=True)
            Q = np.exp(Q / (epsilon + 1e-6))
            action_prob = Q / np.sum(Q, axis=-1, keepdims=True)
            action = np.array([np.random.choice(np.arange(len(action_prob[a])), p=action_prob[a])
                               for a in np.arange(action_prob.shape[0])])

        if get_log:
            return action.reshape(-1, 1), Q
        else:
            return action.reshape(-1, 1)  # n_agents, 1


########################################################################################################################
class ExperienceBuffer(object):
    # N*S, N*A, N*R, N*S', N*doneで管理する．
    def __init__(self, max_size=10**4):
        self.max_size = max_size
        self.reset()


    def reset(self):
        self.state_current = None
        self.action = None
        self.reward = None
        self.state_next = None
        self.done = None

        self.idx = 0  # index on buffer.


    def get_batch(self, batch_size=512):
        if len(self) <= self.idx:  # idxが終端に達している
            self.shuffle()
            self.idx = 0
        state_current = [state[self.idx: self.idx + batch_size] for state in self.state_current]
        state_next = [state[self.idx: self.idx + batch_size] for state in self.state_next]
        action = self.action[self.idx: self.idx + batch_size]
        reward = self.reward[self.idx: self.idx + batch_size]
        done = self.done[self.idx: self.idx + batch_size]

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
        idx = np.random.permutation(len(self.state_current[0]))
        self.state_current = [state[idx] for state in self.state_current]
        self.state_next = [state[idx] for state in self.state_next]
        self.action = self.action[idx]
        self.reward = self.reward[idx]
        self.done = self.done[idx]


    def append(self, experience):
        # experiences = n*S, n*A, n*R, n*S', n*done
        state_current, action, reward, state_next, done = experience
        if len(self) == 0:
            self.state_current = [state_ex for state_ex in state_current]
            self.state_next = [state_ex for state_ex in state_next]
            self.action = action
            self.reward = reward
            self.done = done
            return None

        self.state_current = [np.concatenate([state_in, state_ex], axis=0)[-self.max_size:] \
                              for state_in, state_ex in zip(self.state_current, state_current)]
        self.state_next = [np.concatenate([state_in, state_ex], axis=0)[-self.max_size:] \
                           for state_in, state_ex in zip(self.state_next, state_next)]
        self.action = np.concatenate([self.action, action], axis=0)[-self.max_size:]
        self.reward = np.concatenate([self.reward, reward], axis=0)[-self.max_size:]
        self.done = np.concatenate([self.done, done], axis=0)[-self.max_size:]


########################################################################################################################
class Trainer(object):
    def __init__(self, env, agent, exp_buffer=None):
        self.env = env
        self.agent = agent
        self.model = self.agent.model  # keras Q NN.
        self.model_target = models.Model.from_config(self.model.get_config())  # 同一フォームのモデルを生成

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
                        'info':[],
                        'Q':[]}
        self.log_eva = {'state_current': [],
                        'action': [],
                        'reward': [],
                        'state_next': [],
                        'done': [],
                        'info': [],
                        'Q': []}


    def train(self, n_steps=10000, training_interval=100, n_batches=10, batch_size=256,
              alpha=1.0, target_update_interval=1000,
              doubleQ=False,
              gamma=1.0,
              optimizer=optimizers.Adam(1e-2),
              epsilon_start=1.0, epsilon_end=0.1, epsilon_interval=10000,
              boltzmann=False,
              warmup_steps=500,
              verbose=True, verbose_interval=100, evaluate_interval=1000,
              get_log=False):
        '''

        :param n_steps:
        :param training_interval: 何ステップ数毎に学習を行うか．
        :param n_batches: batch_sizeによるネットワーク更新を何回行うか．
        :param batch_size:
        :param alpha: TDブートストラップ係数α
        :param target_update_interval: ターゲットネットワークの更新頻度．
        :param gamma: 割引係数
        :param optimizer:
        :param epsilon_start: 探索エプシロンの初期値
        :param epsilon_end: 探索エプシロンの最終地
        :param epsilon_interval: 探索エプシロンが最終値に到達するまでのステップ数
        :param verbose:
        :param verbose_interval:
        :param evaluate_interval:
        :return:
        '''

        self.model.compile(optimizer=optimizer, loss='mse')

        self.exp_buffer.reset()
        self.reset_history()
        self.reset_log()
        loss = None

        for t in tqdm(range(n_steps)):
            epsilon = max(epsilon_end, epsilon_start + (epsilon_end - epsilon_start) / epsilon_interval * t)
            s_current = self.env.state()
            if get_log:
                action, Q = self.agent.get_action(s_current, epsilon, boltzmann, greedy=False, get_log=get_log)
            else:
                action = self.agent.get_action(s_current, epsilon, boltzmann, greedy=False, get_log=get_log)
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
                self.log_exp['Q'].append(Q)

            # update
            if (t+1) % training_interval == 0 and warmup_steps<t:
                loss = self._fit_on_batch(n_batches, batch_size, alpha, gamma, doubleQ)
                self.loss_history['all'].append(loss)

            if (t+1) % target_update_interval == 0:
                self.model_target.set_weights(self.model.get_weights())  # 重みをコピペ

            if verbose and (t+1) % verbose_interval == 0 and loss is not None:
                print('step = {}, loss = {:.5f}, epsilon = {:.5f}' \
                    .format(t, loss, epsilon))

            if (t+1) % evaluate_interval==0:
                self.return_history['all'].append(self.evaluate(show_log=verbose, gamma=gamma, get_log=get_log))

        # lossをNDArrayに変換
        self.loss_history = {key:np.array(self.loss_history[key]) for key in self.loss_history.keys()}
        self.return_history = {key:np.array(self.return_history[key]) for key in self.return_history.keys()}

        # log統合
        if get_log:
            for key in ['state_current', 'state_next']:
                pass
                self.log_exp[key] = [np.concatenate(S, axis=0) for S in zip(*self.log_exp[key])]
                self.log_eva[key] = [np.concatenate(S, axis=0) for S in zip(*self.log_eva[key])]

            for key in ['action', 'reward', 'done', 'info', 'Q']:
                self.log_exp[key] = np.concatenate(self.log_exp[key], axis=0)
                self.log_eva[key] = np.concatenate(self.log_eva[key], axis=0)


    def reset_history(self):
        self.loss_history = {key:[] for key in self.loss_history.keys()}
        self.return_history = {key:[] for key in self.return_history.keys()}


    def reset_log(self):
        self.log_exp = {key:[] for key in self.log_exp.keys()}
        self.log_eva = {key:[] for key in self.log_eva.keys()}


    def _fit_on_batch(self, n_batches, batch_size, alpha, gamma, doubleQ):
        '''
        batch_sizeの訓練セットを，minibatch_sizeに切り分けて訓練する，ということをn_iterations回行う．
        '''
        if len(self.exp_buffer) == 0:
            return None

        losses = []
        for i in range(n_batches):
            self.exp_buffer.shuffle()
            S_c, A, R, S_n, done = self.exp_buffer.get_batch(batch_size)

            # target形成(R + gamma * max_a(Q(S',a)))
            Q_pred_curr = self.model_target.predict([*S_c])  # SにおけるQ

            if doubleQ==False:
                Q_pred_next = np.max(self.model_target.predict([*S_n]), axis=1).reshape(-1, 1)  # S'におけるmax Q
            else:
                max_action = np.argmax(self.model.predict([*S_n]), axis=1)
                Q_pred_next = self.model_target.predict([*S_n])[np.arange(max_action.shape[0]), max_action].reshape(-1, 1)  # S'におけるmax Q

            target = np.copy(Q_pred_curr)
            Q_pred_next[done == True] = 0.0  # S'が終端状態である場合はQ(S')は0で上書き
            target[np.arange(done.shape[0]), A[:,0]] = R.ravel() + gamma * Q_pred_next.ravel()  # 実際に選択された行動の部分のみ更新
            target = Q_pred_curr + alpha * (target - Q_pred_curr)

            loss = self.model.train_on_batch([*S_c], target)  # 学習
            losses.append(loss)

        return np.array(losses).mean()


    def evaluate(self, show_log=False, time_horizon=1000, n_showing_agents=10, gamma=1.0,
                 get_log=False):
        self.env.reset()
        rewards = []
        done_mask = 1  # ループ中にNDArrayでブロードキャストされるので，これで問題無い．
        for t in range(time_horizon):
            state_current = self.env.state()
            if get_log:
                action, Q = self.agent.get_action(state_current, greedy=True, get_log=get_log)
            else:
                action = self.agent.get_action(state_current, greedy=True, get_log=get_log)
            state_next, reward, info, done = self.env.step(action)
            rewards.append(done_mask * reward)
            done_mask = done_mask * (1-np.array(done, dtype=int)) * gamma

            # log
            if get_log:
                self.log_eva['state_current'].append(state_current)
                self.log_eva['action'].append(action)
                self.log_eva['reward'].append(reward)
                self.log_eva['state_next'].append(state_next)
                self.log_eva['done'].append(done)
                self.log_eva['info'].append(info)
                self.log_eva['Q'].append(Q)  # DQN specific

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
    def __init__(self, env, model, exp_buffer_size=10 ** 4, name='tester_DQN'):
        self.env = env
        self.model = model
        self.agent = Agent(self.model)
        self.exp_buffer = ExperienceBuffer(exp_buffer_size)
        self.trainer = Trainer(self.env, self.agent, self.exp_buffer)

        self.name = name


    def test(self, n_trials=10, n_steps=10000,
             training_interval=100, n_batches=10, batch_size=256, alpha=0.1, target_update_interval=1000, doubleQ=False,
             gamma=1.00,
             optimizer=optimizers.RMSprop(),
             epsilon_start=1.0, epsilon_end=0.5, epsilon_interval=100, boltzmann=False,
             warmup_steps=500,
             verbose=False, verbose_interval=100, evaluate_interval=100,
             get_log=False, save_objects=False):
        '''

        :param n_trials:
        :param n_steps:
        :param training_interval:
        :param n_batches:
        :param batch_size:
        :param alpha:
        :param target_update_interval:
        :param doubleQ:
        :param gamma:
        :param optimizer:
        :param epsilon_start:
        :param epsilon_end:
        :param epsilon_interval:
        :param boltzmann:
        :param verbose:
        :param verbose_interval:
        :param evaluate_interval:
        :param get_log:
        :return:
        '''

        print('start testing... : ', datetime.now())
        self.loss_histories = []
        self.return_histories = []
        for trial in range(n_trials):
            print('start trial {}/{} trial...'.format(trial + 1, n_trials))
            self.trial(n_steps,
                       training_interval, n_batches, batch_size, alpha, target_update_interval, doubleQ,
                       gamma, optimizer,
                       epsilon_start, epsilon_end, epsilon_interval, boltzmann,
                       warmup_steps,
                       verbose, verbose_interval, evaluate_interval,
                       get_log)

        print('end testing... : ', datetime.now())
        self.report(gamma=gamma)

        # 保存処理
        with open(str(self.name) + '.pkl', 'wb') as f:
            if save_objects:
                pickle.dump([self.trainer , self.loss_histories, self.return_histories], f)
            else:
                pickle.dump([self.loss_histories, self.return_histories], f)


    def trial(self, n_steps,
              training_interval, n_batches, batch_size, alpha, target_update_interval, doubleQ,
              gamma,
              optimizer,
              epsilon_start, epsilon_end, epsilon_interval, boltzmann,
              warmup_steps,
              verbose, verbose_interval, evaluate_interval,
              get_log):

        # modelのリセット
        config = self.model.get_config()
        self.model = models.Model.from_config(config)
        self.agent = Agent(self.model)
        self.trainer = Trainer(self.env, self.agent, self.exp_buffer)  # exp_bufferはtrainer側でリセットされる．

        # optimizerのリセット
        config = optimizer.get_config()
        optimizer = optimizer.from_config(config)

        # 学習
        self.trainer.train(n_steps=n_steps,
                           training_interval=training_interval, n_batches=n_batches, batch_size=batch_size, alpha=alpha,
                           target_update_interval= target_update_interval, doubleQ=doubleQ,
                           gamma=gamma,
                           optimizer=optimizer,
                           epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_interval=epsilon_interval,
                           boltzmann=boltzmann,
                           warmup_steps=warmup_steps,
                           verbose=verbose, verbose_interval=verbose_interval, evaluate_interval=evaluate_interval,
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


    def rep_loss_reward(self, figsize=(15, 7), alpha=None, fontsize=16, dpi=150):
        print('loss and reward history')

        if alpha is None:
            alpha = np.sqrt(1 / len(self.loss_histories))

        fig = plt.figure(figsize=figsize, dpi=dpi)

        ax = fig.add_subplot(1, 2, 1)
        for i, loss_history in enumerate(self.loss_histories):
            for key in loss_history.keys():
                ax.plot(loss_history[key], c=self.trainer.loss_color[key], alpha=alpha, label=key if i == 0 else None)  # black
        ax.set_title('loss history', fontsize=fontsize)
        ax.set_xlabel('data step', fontsize=fontsize)
        ax.set_ylabel('loss', fontsize=fontsize)
        ax.legend(fontsize=fontsize)

        ax = fig.add_subplot(1, 2, 2)
        for i, return_history in enumerate(self.return_histories):
            for key in return_history.keys():
                ax.plot(return_history[key], c=self.trainer.return_color[key], alpha=alpha, label=key if i == 0 else None)  # black
        ax.set_title('return history', fontsize=fontsize)
        ax.set_xlabel('data step', fontsize=fontsize)
        ax.set_ylabel('return', fontsize=fontsize)

        ax.legend(fontsize=fontsize)

        plt.tight_layout()
        plt.show()


    def rep_action(self, gamma=1.0):
        print('acquired action')
        self.trainer.evaluate(show_log=True, n_showing_agents=10, gamma=gamma, get_log=False)


    def rep_param(self, figsize=(15, 5), dpi=150):
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


    def viz_action_history(self, trainer, state_valuation, data_span=10,
                           figsize=(15, 7), dpi=150, cmap='jet', fontsize=16, alpha=0.1, s=0.1):
        '''

        :param trainer: Trainer instance contains log
        :param state_valuation: function that maps state to number
        :return:
        '''
        def log_get(log):
            state_current = state_valuation(log['state_current'])
            action = log['action']

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


    def viz_Q_history(self, trainer, data_span_exp=10, data_span_eva=10,
                      figsize=(15, 5), dpi=150, cmap='jet', fontsize=16):
        plt.figure(figsize=figsize, dpi=dpi)
        sns.heatmap(trainer.log_exp['Q'][::data_span_exp].T, cmap=cmap)
        plt.title('evolution of Q in exploration', size=fontsize)
        plt.xlabel('data step', size=fontsize)
        plt.ylabel('action', size=fontsize)
        plt.show()

        plt.figure(figsize=figsize, dpi=dpi)
        sns.heatmap(trainer.log_eva['Q'][::data_span_eva].T, cmap=cmap)
        plt.title('evolution of Q in evaluation', size=fontsize)
        plt.xlabel('data step', size=fontsize)
        plt.ylabel('action', size=fontsize)
        plt.show()


########################################################################################################################

if __name__ == '__main__':
    n_agents = 4
    state_dims = [1, 3]
    n_outputs = 10
    env = Environment(n_agents=n_agents, state_dims=state_dims, n_outputs=n_outputs)
    print(env.state())
    print(env.step(0))
    print(env.step(1))

    model = gen_model(input_shapes=[[d, ] for d in state_dims], n_outputs=n_outputs, hidden_dims=[2, 4, 6])

    start_time = time.time()
    trainer = Trainer(env, Agent(model))
    trainer.train(verbose=False)
    print(time.time() - start_time)


    print('END')