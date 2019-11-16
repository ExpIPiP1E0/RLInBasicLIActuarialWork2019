import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import layers
from keras import regularizers
from keras import models

from tqdm import tqdm



'''
GA (Deep Neural Evolution)

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
        reward = np.random.randint(-1, 1+1, self.done.shape)
        self.done = np.random.binomial(1, 0.2, self.done.shape)

        return [self.state(), np.copy(reward), None, np.copy(self.done)]


    def state(self):
        return [np.copy(state) for state in self.states]


    def reset(self):
        self.states = []
        for state_dim in self.state_dims:
            self.states.append(np.arange(state_dim) * np.ones(self.n_agents).reshape(-1, 1))
        self.done = np.zeros(shape=(self.n_agents, 1), dtype=bool)


########################################################################################################################
def gen_model(input_shapes=[[10], [20]], n_outputs=10,
              hidden_dims=[512, 256, 128, 64, 32], reg_l1=0.0, reg_l2=0.0,
              action_min=0.0, action_max=0.0,
              input_reg=False, input_min=-10, input_max=10):
    input_ts = [layers.Input(input_shape) for input_shape in input_shapes]
    if 2<=len(input_ts):
        input_concat = layers.concatenate(input_ts, axis=-1)
    else:
        input_concat = layers.Lambda(lambda x: x)(input_ts[0])  # 恒等レイヤー

    if input_reg:
        input_concat = layers.Lambda(lambda x: x/(input_max-input_min))(input_concat)

    for i, hidden_dim in enumerate(hidden_dims):
        if i==0:
            x = layers.LeakyReLU(alpha=0.1)(layers.Dense(hidden_dim, kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(input_concat))
        else:
            x = layers.LeakyReLU(alpha=0.1)(layers.Dense(hidden_dim, kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x))

    action = layers.Dense(n_outputs, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(x)
    action = layers.Lambda(lambda x: action_min + (action_max-action_min) * x, name='action')(action)

    return models.Model(input_ts, action)


########################################################################################################################
class Agent(object):
    '''
    GAの場合，Agentは必ずしも必要では無いが，API体系を保持するため残している．
    '''
    def __init__(self, model):
        self.model = model  # Policy Network


    def get_action(self, state, greedy=False, get_log=False):
        # greedy and get_log are dummy input.
        return self.model.predict([*state])  # n_agents, n_state_dims


########################################################################################################################
class Trainer(object):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.model = agent.model

        self.loss_history = {'all':[]}
        self.return_history = {'all':[]}
        self.loss_color = {'all':'r'}  # for graph
        self.return_color = {'all':'b'}  # for graph

        self.log_exp = {'state_current': [],
                        'action': [],
                        'reward': [],
                        'state_next': [],
                        'done': [],
                        'info': []}
        self.log_eva = {'state_current': [],
                        'action': [],
                        'reward': [],
                        'state_next': [],
                        'done': [],
                        'info': []}


    def train(self, n_episodes=1000, n_samples=256, ratio_elite=0.5,
              gamma=1.0,
              epsilon_start=1.0, epsilon_end=0.1, epsilon_interval=100,
              verbose=True, verbose_interval=100, evaluate_interval=1000,
              get_log=False):

        self.reset_history()
        loss = None
        self.samples = [[self.evaluate(gamma=gamma), self.model.get_weights()]]  # initial

        # sampling
        for e in tqdm(range(n_episodes)):
            epsilon = self._rate(epsilon_start, epsilon_end, epsilon_interval, e)
            loss = self._fit(n_samples, gamma, epsilon, get_log)

            # choose elite
            self.samples.sort(key=lambda x: x[0], reverse=True)
            self.samples = self.samples[: int(1 + n_samples * ratio_elite)]
            self.model.set_weights(self.samples[0][1])  # evaluation用

            self.loss_history['all'].append(loss)

            if verbose and e % verbose_interval==0 and loss is not None:
                print('episode = {}, loss={:.5f}, epsilon = {:.5f}' \
                      .format(e, loss, epsilon))

            if (e + 1) % evaluate_interval == 0:
                self.return_history['all'].append(
                    self.evaluate(show_log=verbose, gamma=gamma, get_log=get_log, log_type_exp=False))

        # lossをNDArrayに変換
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


    def reset_history(self):
        self.loss_history = {key: [] for key in self.loss_history.keys()}
        self.return_history = {key: [] for key in self.return_history.keys()}


    def reset_log(self):
        self.log_exp = {key:[] for key in self.log_exp.keys()}
        self.log_eva = {key:[] for key in self.log_eva.keys()}


    def _rate(self, rate_start, rate_end, rate_interval, current_step):
        return np.clip(rate_start + (rate_end - rate_start) / rate_interval * current_step,
                       min(rate_start, rate_end), max(rate_start, rate_end))


    def _fit(self, n_samples, gamma, epsilon,
             get_log=False):

        # get samples
        samples = [self.samples[0], ]  # 現時点のトップは必ず残す．
        reward_pre = sum([s[0] for s in self.samples]) / len(self.samples)

        for s in range(n_samples):
            # generate noise
            noise_pos = [epsilon * np.random.randn(*weight.shape) for weight in self.model.get_weights()]
            noise_neg = [-1*noise for noise in noise_pos]

            # generate models
            original_weights = self.samples[np.random.randint(len(self.samples))][1]
            weights_pos = [weight + noise for weight, noise in zip(noise_pos, original_weights)]
            weights_neg = [weight + noise for weight, noise in zip(noise_neg, original_weights)]

            # generate experience and store
            self.model.set_weights(weights_pos)
            reward = np.mean(self.evaluate(gamma=gamma, get_log=get_log, log_type_exp=True))
            samples.append([reward, weights_pos])

            self.model.set_weights(weights_neg)
            reward = np.mean(self.evaluate(gamma=gamma, get_log=get_log, log_type_exp=True))
            samples.append([reward, weights_neg])

        self.samples = samples  # update
        reward_post = sum([s[0] for s in self.samples]) / len(self.samples)

        return reward_post - reward_pre


    def evaluate(self, show_log=False, time_horizon=1000, n_showing_agents=10, gamma=1.0,
                 get_log=False, log_type_exp=True):
        # これまでと違い，evaluateはtrainそのものにも使用されるので注意．
        self.env.reset()
        rewards = []
        done_mask = 1
        for t in range(time_horizon):
            state_current = self.env.state()
            action = self.agent.get_action(state_current, greedy=True, get_log=get_log)
            state_next, reward, info, done = self.env.step(action)
            rewards.append(done_mask * reward)
            done_mask = done_mask * (1-np.array(done, dtype=int)) * gamma

            # log
            if get_log:
                if log_type_exp:
                    log = self.log_exp
                else:
                    log = self.log_eva
                log['state_current'].append(state_current)
                log['action'].append(action)
                log['reward'].append(reward)
                log['state_next'].append(state_next)
                log['done'].append(done)
                log['info'].append(info)

            if show_log:
                print('action = ', action.ravel()[:n_showing_agents])

            if done_mask.sum() == 0:
                break

        rewards = np.array(rewards)[:, :, 0]  # T, n_agents, 1

        if show_log:
            print('total_reward', rewards.sum(axis=0)[:n_showing_agents])

        return np.mean(rewards.sum(axis=0))  # n_agents


########################################################################################################################
from datetime import datetime
import pickle

class Tester(object):
    def __init__(self, env, model, name='tester_GA'):
        self.env = env
        self.model = model
        self.agent = Agent(self.model)
        self.trainer = Trainer(self.env, self.agent)

        self.name = name


    def test(self, n_trials=5, n_episodes=100,
             n_samples=64, ratio_elite=0.1,
             gamma=1.0,
             epsilon_start=0.01, epsilon_end=0.01, epsilon_interval=100,
             verbose=False, verbose_interval=100, evaluate_interval=100,
             get_log=False, save_objects=False):
        print('start testing... : ', datetime.now())
        self.loss_histories = []
        self.return_histories = []
        for trial in range(n_trials):
            print('start trial {}/{} trial...'.format(trial, n_trials))
            self.trial(n_episodes, n_samples, ratio_elite,
                       gamma,
                       epsilon_start, epsilon_end, epsilon_interval,
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


    def trial(self, n_episodes, n_samples, ratio_elite,
              gamma,
              epsilon_start, epsilon_end, epsilon_interval,
              verbose, verbose_interval, evaluate_interval,
              get_log):

        # modelのリセット
        config = self.model.get_config()
        self.model = models.Model.from_config(config)
        self.agent = Agent(self.model)
        self.trainer = Trainer(self.env, self.agent)

        # 学習
        self.trainer.train(n_episodes=n_episodes,
                           n_samples=n_samples, ratio_elite=ratio_elite,
                           gamma=gamma,
                           epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_interval=epsilon_interval,
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


    def rep_param(self, figsize=(15, 5), fontsize=10):
        print('param distribution in model')
        p_layers = [layer for layer in self.model.layers if len(layer.get_weights())!=0]

        fig = plt.figure(figsize=figsize)
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


########################################################################################################################
if __name__=='__main__':
    n_agents = 4
    state_dims = [1, 3]
    n_outputs = 1

    env = Environment(n_agents=n_agents, state_dims=state_dims, n_outputs=n_outputs)
    print(env.state())
    print(env.step(np.array([[0, ]])))
    print(env.step(np.array([[1, ]])))

    model = gen_model(input_shapes=[[d, ] for d in state_dims], n_outputs=n_outputs, hidden_dims=[2, 4, 6])
    agent = Agent(model)
    agent.get_action(env.state())
    trainer = Trainer(env, agent)
    tester = Tester(env, model)
    tester.test()


    print('END')


