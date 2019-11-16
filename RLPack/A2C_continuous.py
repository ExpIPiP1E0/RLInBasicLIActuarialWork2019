
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras import layers
from keras import models
from keras import regularizers
from keras import optimizers
import keras.backend as K

import math
from tqdm import tqdm
from datetime import datetime
import pickle



'''
Continuous version of A2C.
DNN returns Gaussiann distribution parameter of the policy.
例えば，N次元の行動空間の場合，N次元のガウス分布のパラメータ，即ちN×2のパラメータを返す（とりま共分散は無視）．
また，このとき，この方策のエントロピーはlog √(2πeσ^2)である．
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
            #state += (action==1) * np.random.uniform(size=state.shape)  # *= で入れないと無効．
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
def gen_model(input_shapes=[[10], [20]], n_outputs=3, \
              hidden_dims=[2, 4, 6], reg_l1=0.0, reg_l2=0.0, \
              mu_min=0.00, mu_max=10.0, va_min=0.0, va_max=10.0,
              input_reg=False, input_min=-10, input_max=10):
    '''
    :param input_shapes: list of input shapes.
    :param n_outputs: specify output action dimension.
    :param hidden_dims: list of newrons in each hidden layers.
    :return: probability of action, value
    neural net generator for function approximation of actor-critic policy and value.
    '''
    input_ts = [layers.Input(input_shape) for input_shape in input_shapes]
    if 2<=len(input_ts):
        input_concat = layers.concatenate(input_ts, axis=-1)
    else:
        input_concat = layers.Lambda(lambda x: x)(input_ts[0])  # 恒等レイヤー．

    if input_reg:
        input_concat = layers.Lambda(lambda x: (x - input_min) / (input_max - input_min))(input_concat)

    for i, hidden_dim in enumerate(hidden_dims):
        if i==0:
            x = layers.LeakyReLU(alpha=0.1)(layers.Dense(hidden_dim, kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(input_concat))
        else:
            x = layers.LeakyReLU(alpha=0.1)(layers.Dense(hidden_dim, kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x))

    va_min = max(1e-8, va_min)
    va_max = max(1e-8, va_min, va_max)

    policy_mu_head = layers.Dense(n_outputs, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x)
    policy_mu_head = layers.Lambda(lambda x: mu_min + (mu_max-mu_min) * x, name='policy_mu')(policy_mu_head)
    policy_va_head = layers.Dense(n_outputs, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x)
    policy_va_head = layers.Lambda(lambda x: va_min + (va_max-va_min) * x, name='policy_va')(policy_va_head)

    policy_head = layers.concatenate([policy_mu_head, policy_va_head], axis=-1, name='policy_head')
    value_head = layers.Dense(1, activation='linear', name='value_head')(x)

    return models.Model(input_ts, [policy_head, value_head])


########################################################################################################################
class Agent(object):
    def __init__(self, model, val_min=-np.inf, val_max=np.inf):
        '''
        val_min and val_max affect policy.
        not recommend to use.
        :param model:
        :param val_min:
        :param val_max:
        '''
        self.model = model
        self.val_min = val_min
        self.val_max = val_max


    def get_action(self, state, greedy=False, get_log=False):
        # state.shape = dim_state * n_agents * dim_each_state
        ps, value = self.model.predict([*state])  # ps.shape = 3 * N * n_actions
        mu = ps[:, :ps.shape[-1]//2] ; va = ps[:, ps.shape[-1]//2:]  # N, n_actions
        Z = np.random.randn(*mu.shape) * (0 if greedy else 1)
        if get_log:
            return np.clip(mu + np.sqrt(va) * Z, self.val_min, self.val_max), mu, va, value
        else:
            return np.clip(mu + np.sqrt(va) * Z, self.val_min, self.val_max)


########################################################################################################################
class Trainer(object):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.model = agent.model
        #self.model.compile(loss='mse', optimizer='sgd')

        # 学習用に出力だけ結合したモデルを作成する．
        output_concat = layers.concatenate([layer for layer in self.model.outputs], axis=1)
        self.model_train = models.Model(self.model.inputs, output_concat)

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
                        'va': [],
                        'value': []}
        self.log_eva = {'state_current': [],
                        'action': [],
                        'reward': [],
                        'state_next': [],
                        'done': [],
                        'info': [],
                        'mu': [],
                        'va': [],
                        'value': []}


    def train(self, n_steps, n_advantages,
              gamma,
              optimizer,
              mult_actor=1.0, mult_critic=1.0, mult_entropy=0.1,
              verbose=True, verbose_interval=100, evaluate_interval=1000,
              get_log=False):
        self.mult_actor = mult_actor
        self.mult_critic = mult_critic
        self.mult_entropy = mult_entropy

        self.model_train.compile(optimizer=optimizer, loss=self.loss_all,
                                 metrics=[self.loss_actor, self.loss_critic, self.loss_entropy])
        self.reset_buffer()
        self.reset_history()
        self.reset_log()
        t_current = np.zeros(shape=(self.env.state()[0].shape[0], 1))
        loss = None

        batch_count = 0
        for t in tqdm(range(n_steps)):
            state_current = self.env.state()
            if get_log:
                action, mu, va, value = self.agent.get_action(state_current, greedy=False, get_log=get_log)
            else:
                action = self.agent.get_action(state_current, greedy=False, get_log=get_log)
            state_next, reward, info, done = self.env.step(action)

            self.state_current.append(state_current)  # n_advantages, n_types, n_agents, n_state_dims
            self.state_next.append(state_next)
            self.action.append(action)
            self.reward.append(reward)
            self.done.append(done)
            self.t_current.append(t_current)
            batch_count += 1

            # update
            if batch_count == n_advantages:
                loss = self._fit_on_batch(gamma)
                self.reset_buffer()  # clear self.state_current, state_next, action, reward, done

                self.loss_history['all'].append(loss[0])
                self.loss_history['actor'].append(loss[1])
                self.loss_history['critic'].append(loss[2])
                self.loss_history['entropy'].append(loss[3])
                batch_count = 0

            t_current = (t_current + 1) * (1-done)

            # log
            if get_log:
                self.log_exp['state_current'].append(state_current)
                self.log_exp['action'].append(action)
                self.log_exp['reward'].append(reward)
                self.log_exp['state_next'].append(state_next)
                self.log_exp['done'].append(done)
                self.log_exp['info'].append(info)
                self.log_exp['mu'].append(mu)
                self.log_exp['va'].append(va)
                self.log_exp['value'].append(value)

            if verbose and (t+1) % verbose_interval==0 and loss is not None:
                print('step = {}, loss_all = {:.5f}, loss_actor = {:.5f}, loss_critic = {:.5f}, loss_entropy = {:.5f}' \
                    .format(t, loss[0], loss[1], loss[2], loss[3]))

            if (t+1) % evaluate_interval==0:
                self.return_history['all'].append(self.evaluate(show_log=verbose, gamma=gamma, get_log=get_log))
                self.reset_buffer()  # evaluateの際にenv.resetがかかり不整合になるので，情報を捨てる…done以外でコールできないので捨てるしか無い．
                batch_count = 0
                t_current = t_current * 0

        # lossをNDArrayに変換
        self.loss_history = {key: np.array(self.loss_history[key]) for key in self.loss_history.keys()}
        self.return_history = {key: np.array(self.return_history[key]) for key in self.return_history.keys()}

        # log統合
        if get_log:
            for key in ['state_current', 'state_next']:
                pass
                self.log_exp[key] = [np.concatenate(S, axis=0) for S in zip(*self.log_exp[key])]
                self.log_eva[key] = [np.concatenate(S, axis=0) for S in zip(*self.log_eva[key])]

            for key in ['action', 'reward', 'done', 'info', 'mu', 'va', 'value']:
                self.log_exp[key] = np.concatenate(self.log_exp[key], axis=0)
                self.log_eva[key] = np.concatenate(self.log_eva[key], axis=0)


    def reset_buffer(self):
        self.state_current = []
        self.state_next = []
        self.action = []
        self.reward = []
        self.done = []
        self.t_current = []


    def reset_history(self):
        self.loss_history = {key:[] for key in self.loss_history.keys()}
        self.return_history = {key:[] for key in self.return_history.keys()}


    def reset_log(self):
        self.log_exp = {key:[] for key in self.log_exp.keys()}
        self.log_eva = {key:[] for key in self.log_eva.keys()}


    def _fit_on_batch(self, gamma):
        # experienceは，advantage_id, attribute_type, agent_id, attribute_dimのインデックスを持っている．
        # これを一旦，advantage_id * agent_idの部分まではバッチ化してmodelに入力できるようにする(state)．

        # convert experiences to batch for self.model
        n_advantages = len(self.state_current)

        state_current = []
        state_next = []
        for state_type in range(len(self.state_current[0])):
            state_current.append([])
            state_next.append([])
            for k in range(n_advantages):
                # state_current.shape = n_types, n_advantages, n_agents, n_state_dims
                state_current[state_type].append(self.state_current[k][state_type])  # n_advantages, n_types, n_agents, n_state_dims
                state_next[state_type].append(self.state_next[k][state_type])
            state_current[state_type] = np.concatenate([*(state_current[state_type])], axis=0)
            state_next[state_type] = np.concatenate([*(state_next[state_type])], axis=0)
        reward = np.array([R[:, 0] for R in self.reward])  # n_advantages, n_agents, 1
        done = np.array([D[:, 0] for D in self.done], dtype=int)  # n_advantages, n_agents, 1

        P_current, V_current = self.model.predict(state_current)
        P_next, V_next = self.model.predict(state_next)

        #P_current, P_next = [P.reshape() for P in [P_current, P_next]]
        V_current, V_next = [V.reshape(n_advantages, -1) for V in [V_current, V_next]]

        # create target for V
        # 最後尾データから戻していく．但し，途中で終端状態が入っていればそれ以前には引き継がない．
        target = []
        for k in range(n_advantages)[::-1]:  # 逆側から開始
            if k==n_advantages-1:  # 最初
                target.append(reward[k, :] + gamma * V_next[k, :] * (1-done[k, :]))
            else:
                target.append(reward[k, :] + gamma * target[-1] * (1-done[k, :]))
        target = list(reversed(target))
        target = np.array(target).reshape(-1, 1)

        # model学習
        action = np.concatenate([A[:, :] for A in self.action], axis=0)  # n_advantages, n_agents, n_actions
        #action = np.eye(P_current.shape[-1])[action.ravel()]  # one-hotに変換．
        y_true = np.concatenate([action, target], axis=1)
        t_current = np.array(self.t_current)[:, :, 0].ravel()
        disc_fac = gamma ** t_current

        return self.model_train.train_on_batch(state_current, y_true, sample_weight=disc_fac)  # lossを返す．


    def loss_all(self, y_true, y_pred):
        '''
        A2C has 3 loss functions.
        1.Policy loss : log policy(A|S)(T-V) with keep R-V constant.
        2.Value loss : (T-V)^2
        3.Entropy loss for regularization.
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
        action_prob_va = action_prob[:, K.int_shape(action_prob)[-1]//2:]
        v_target = y_true[:, -1:]  # n-step V at action a
        v_pred = y_pred[:, -1:]  # V pred by model

        p1 = ((action_prob_mu - action)**2) / (2 * action_prob_va + K.epsilon())
        p2 = K.log(K.sqrt(2 * math.pi * action_prob_va))

        return self.mult_actor * (p1 + p2) * K.stop_gradient(v_target-v_pred)


    def loss_critic(self, y_true, y_pred):
        v_target = y_true[:, -1:]  # n-step V at action a
        v_pred = y_pred[:, -1:]  # V pred by model
        return self.mult_critic * K.sum(K.square(v_target-v_pred), axis=-1, keepdims=True)


    def loss_entropy(self, y_true, y_pred):
        action_prob = y_pred[:, :-1]
        action_prob_va = action_prob[:, K.int_shape(action_prob)[-1]//2:]  # N * n_actions
        return self.mult_entropy * K.sum((-(K.log(2 * math.pi * action_prob_va)+1))/2, axis=-1, keepdims=True)


    def evaluate(self, show_log=False, time_horizon=1000, n_showing_agents=10, gamma=1.0,
                 get_log=False):
        self.env.reset()
        rewards = []
        done_mask = 1  # ループ中にNDArrayでブロードキャストされるので，これで問題無い．
        for t in range(time_horizon):
            state_current = self.env.state()
            if get_log:
                action, mu, va, value = self.agent.get_action(state_current, greedy=True, get_log=get_log)
            else:
                action = self.agent.get_action(state_current, greedy=True, get_log=get_log)
            state_next, reward, info, done = self.env.step(action)
            rewards.append(done_mask * reward)  # apply done_mask for cut off already done agents
            done_mask = done_mask * (1-np.array(done, dtype=int)) * gamma  # update done_mask

            # log
            if get_log:
                self.log_eva['state_current'].append(state_current)
                self.log_eva['action'].append(action)
                self.log_eva['reward'].append(reward)
                self.log_eva['state_next'].append(state_next)
                self.log_eva['done'].append(done)
                self.log_eva['info'].append(info)
                self.log_eva['mu'].append(mu)
                self.log_eva['va'].append(va)
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
    def __init__(self, env, model, name='tester_A2C', val_min=0.0, val_max=10.0):
        self.env = env
        self.model = model
        self.agent = Agent(model, val_min=val_min, val_max=val_max)
        self.trainer = Trainer(env, self.agent)

        self.name = name  # this attribute is used for reporting, saving results.


    def test(self, n_trials=5, n_steps=1000,
             n_advantages=4,
             gamma=1.0,
             optimizer=optimizers.Adam(1e-2),
             mult_actor=1.0, mult_critic=1.0, mult_entropy=0.1,
             verbose=False, verbose_interval=100, evaluate_interval=100,
             get_log=False, save_objects=False):
        '''
        :param n_trials: number of trials for testing.
        :param n_steps: number of steps for training for a trial.
        :param n_advantages: advantages for calculating target value calculation.
        :param gamma: discount factor.
        :param lr: learning rate.
        :param mult_entropy: mult for entropy loss function.
        :param verbose: on/off verbose.
        :param verbose_interval: step interval for verbose.
        :param evaluate_interval: step interval for evaluating model (greedy action and score)
        :return:
        '''
        print('start testing... : ', datetime.now())
        self.loss_histories = []
        self.return_histories = []
        for trial in range(n_trials):
            print('start {}/{} th trial...'.format(trial, n_trials))
            self.trial(n_steps,
                       n_advantages,
                       gamma,
                       optimizer,
                       mult_actor, mult_critic, mult_entropy,
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
              n_advantages,
              gamma,
              optimizer,
              mult_actor, mult_critic, mult_entropy,
              verbose, verbose_interval, evaluate_interval,
              get_log):

        # modelのリセット
        config = self.model.get_config()
        self.model = models.Model.from_config(config)
        self.agent = Agent(self.model, self.agent.val_min, self.agent.val_max)
        self.trainer = Trainer(self.env, self.agent)

        # optimizerのリセット
        config = optimizer.get_config()
        optimizer = optimizer.from_config(config)

        # 学習
        self.trainer.train(n_steps=n_steps,
                           n_advantages=n_advantages,
                           gamma=gamma,
                           optimizer=optimizer,
                           mult_actor=mult_actor, mult_critic=mult_critic, mult_entropy=mult_entropy,
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
    n_agents = 4
    state_dims = [1, 3]
    n_outputs = 1

    env = Environment(n_agents=n_agents, state_dims=state_dims, n_outputs=n_outputs)
    print(env.state())
    print(env.step(np.array([[0,]])))
    print(env.step(np.array([[1,]])))

    model = gen_model(input_shapes=[[d,] for d in state_dims], n_outputs=n_outputs, hidden_dims=[2, 4, 6])
    agent = Agent(model)
    agent.get_action(env.state())
    trainer = Trainer(env, agent)
    tester = Tester(env, model)

    #trainer.train(10)
    tester.test()


    print('END')