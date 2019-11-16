import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fmin  # 最適解を導出するための数理最適化ソルバー

np.set_printoptions(precision=3)  # 数値桁数を指定．

from tqdm import tqdm

########################################################################################################################
class CAPM(object):
    def __init__(self, n_agents=64, returns=[0.01, 0.05],
                 sigmas=[0, 0.1],
                 gen_automatic=False,
                 gen_returns_mu=0.01, gen_returns_sigma=0.05,
                 gen_sigmas_mu=0.001, gen_sigmas_sigma=0.10,
                 action_default_mode='continuous'):
        '''

        :param n_agents: number of agents
        :param returns: mean return of each asset (list)
        :param sigmas: std deviation of each asset (list)
        :param gen_automatic: generate mu and sigma automatically
        :param gen_returns_mu: mean of mu generated
        :param gen_returns_sigma: std of mu generated
        :param gen_sigmas_mu: mean of sigma generated
        :param gen_sigmas_sigma: std of sigma generated
        :param action_default_mode:
        '''

        self.n_agents = n_agents
        self.n_assets = len(returns)
        self.returns = np.tile(np.array(returns).reshape(1, -1), (n_agents, 1))  # n_agents, n_assets
        self.sigmas = np.tile(np.array(sigmas).reshape(1, -1), (n_agents, 1))  # n_agents, n_assets

        self.gen_automatic = gen_automatic
        self.gen_returns_mu = gen_returns_mu
        self.gen_returns_sigma = gen_returns_sigma
        self.gen_sigmas_mu = gen_sigmas_mu
        self.gen_sigmas_sigma = gen_sigmas_sigma

        self.action_default_mode = action_default_mode

        if gen_automatic:
            self.gen_parameters()


    def step(self, action, auto_reset=True,
             return_sigma_mu=False, action_mode=None, uniform=True, verbose=False):
        '''

        :param action:
        :param auto_reset: reset automatically when done
        :param return_sigma_mu: return mu and sigma sepalately
        :param action_mode: if discrete, it is considered as to speciy action id as (n_agents, 1)
        :param uniform: standarize action
        :param verbose:
        :return:
        '''
        # action interpreter when type is not continuous
        if action_mode is None:
            action_mode = self.action_default_mode

        if action_mode == 'discrete':
            # action is considered as 0, 1, 2, 3...K-1
            action = self.convert_action(action)

        if uniform:
            action = action + 1e-6
            action = action / np.sum(action, axis=-1, keepdims=True)

        # action.shape = (n_agents, n_assets), return.shape = (n_agents, n_assets)
        mu = np.sum(self.returns * action, axis=-1, keepdims=True)
        # (1, n_assets) * (n_agents, n_assets) = (n_agents, n_assets)
        sigma = np.sqrt(np.sum((self.sigmas ** 2) * (action ** 2), axis=-1, keepdims=True))

        reward = mu - sigma  # reward is defined as this.
        state_next = self.state()
        info = np.zeros(shape=(self.n_agents, 1), dtype=str)
        done = np.ones_like(info, dtype=bool)  # dummy

        if self.gen_automatic:
            self.gen_parameters()

        if not return_sigma_mu:
            return state_next, reward, info, done
        else:
            return state_next, reward, [mu, sigma], done


    def state(self):
        return [np.copy(self.returns), np.copy(self.sigmas)]


    def reset(self):  # dummy
        self.returns = np.tile(np.array(self.returns[0, :]).reshape(1, -1), (self.n_agents, 1))  # n_agents, n_assets
        self.sigmas = np.tile(np.array(self.sigmas[0, :]).reshape(1, -1), (self.n_agents, 1))  # n_agents, n_assets

        if self.gen_automatic:
            self.gen_parameters()

        return self.state()


    def shapes(self):
        return [S.shape[1:] for S in self.state()]


    def gen_parameters(self):
        self.returns = np.clip(  # n_agents, n_assets
            self.gen_returns_mu + self.gen_returns_sigma * np.random.randn(self.n_agents, self.n_assets), 0, 1)
        self.sigmas = np.clip(
            self.gen_sigmas_mu + self.gen_sigmas_sigma * np.random.randn(self.n_agents, self.n_assets), 0, 1)


    def convert_action(self, action_dis, n_split=10, uniform=True):
        # convert discrete action input to n_agents, n_assets type array.
        action_con = []
        action_dis = action_dis.astype(int)
        for _ in range(self.n_assets):
            action_con.append(action_dis % n_split)  # n_assets, n_agents, 1
            action_dis = action_dis // n_split

        action_con = np.array(action_con, dtype=float)[::-1, :, 0].T

        if uniform:
            action_con = action_con + 1e-6  # adjust for zero devide
            action_con = action_con / np.sum(action_con, axis=-1, keepdims=True)

        return action_con


    # special methods for the environment
    def calc_frontier(self):
        return None

    ####################################################################################
    def show_simulation(self, n_samples=1000):
        self.reset()  # just formal.
        current_action_mode = self.action_default_mode
        self.action_default_mode = 'continuous'

        # データ収集
        actions = []
        mu = []
        sigma = []
        for _ in tqdm(range(n_samples)):
            action = np.random.uniform(size=(self.n_agents, self.n_assets))
            actions.append(np.copy(action))
            action /= np.sum(action, axis=-1, keepdims=True)  # standardize

            mu_, sigma_ = self.step(action, return_sigma_mu=True)[2]
            mu.append(mu_)
            sigma.append(sigma_)

        mu = np.array(mu).ravel()
        sigma = np.array(sigma).ravel()

        # グラフ領域作成
        fig = plt.figure(figsize=(15, 12), dpi=100)

        # sigma vs returnのグラフ
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.scatter(x=sigma, y=mu, c='r', s=0.1, alpha=0.5)
        ax1.set_title('sigma vs return', fontsize=14)
        ax1.set_xlabel('sigma', fontsize=14)
        ax1.set_ylabel('return', fontsize=14)

        # distribution of reward
        actions = np.concatenate(actions, axis=0)
        idx = np.argmax(np.ravel(mu - sigma))
        ax2 = fig.add_subplot(2,
                              2 if self.n_assets==2 else 1,
                              3 if self.n_assets==2 else 2)
        ax2.hist(np.ravel(mu - sigma), bins=20)
        ax2.set_title('distribution of reward (return - sigma)', fontsize=14)
        ax2.set_xlabel('reward', fontsize=14)
        ax2.set_ylabel('samples', fontsize=14)

        # 行動のヒートマップ
        if self.n_assets == 2:
            ax3 = fig.add_subplot(2, 2, 4)
            ax3.scatter(x=actions[:, 0], y=actions[:, 1], c=mu - sigma, cmap='jet')
            ax3.set_title('reward', fontsize=14)
            ax3.set_xlabel('action_0 : volume of asset 0', fontsize=14)
            ax3.set_ylabel('action_1 : volume of asset 1', fontsize=14)

        plt.show()

        print('on sample    : maximum reward={:.6f} at action={}, mu={:.6f}, sigma={:.6f}'
              .format(mu[idx] - sigma[idx], actions[idx], mu[idx], sigma[idx]))
        X = self._normal_optimize()
        print('on optimizer : maximum reward={:.6f} at action={}, mu={:.6f}, sigma={:.6f}'
              .format(X[0], X[1], X[2], X[3]))

        self.action_default_mode = current_action_mode

        # 証券投資理論で学んだような，投資可能範囲・効率的フロンティアを示す散布図が得られる


    def _normal_optimize(self, show_process=False):
        softmax = lambda X: np.exp(X)/ np.sum(np.exp(X))
        portfolio_return = lambda R, rs, ss: -(np.sum(softmax(R) * rs) - np.sqrt(np.sum((softmax(R) ** 2) * (ss ** 2))))
        args = (self.returns[0], self.sigmas[0])

        self.count = 0
        def cbf(X):
            self.count += 1
            f = portfolio_return(X, args[0], args[1])
            print(self.count, X[0], X[1], f)

        result = fmin(portfolio_return, np.zeros(self.n_assets), args=args, disp=show_process, callback=cbf if show_process else None)
        del self.count

        return -portfolio_return(result, *args), softmax(result), \
               np.sum(softmax(result) * self.returns[0]), \
               np.sqrt(np.sum((softmax(result)**2) * (self.sigmas[0]**2)))







