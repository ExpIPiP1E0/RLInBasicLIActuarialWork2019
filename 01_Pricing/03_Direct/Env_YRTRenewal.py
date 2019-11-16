import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=3)  # 数値桁数を指定．


########################################################################################################################
class YRTRenewal(object):
    def __init__(self,
                 n_agents=64,
                 term=20,
                 total_volume=1,
                 prm_min_NB=0 /100, prm_max_NB=10 /100, ratio_NB_max=1.0,
                 prm_min_IF=10 /100, prm_max_IF=20 /100, ratio_IF_max=1.0,
                 initial_cash=300 / 100,
                 expense_fixed=0 /100, expense_variable=0.0 /100, interest=0.00,
                 mult_action=1 /100,
                ):
        '''

        :param n_agents:
        :param term: episode length
        :param total_volume: total market volume
        :param prm_min_NB: premium at when demand for NB becomes 100%
        :param prm_max_NB: premium at when demand for NB becomes 0%
        :param ratio_NB_max: maximum NB ratio at one time when premium is lower than prm_min_NB
        :param prm_min_IF: premium at when demand for IF becomes 100%
        :param prm_max_IF: premium at when demand for IF becomes 0%
        :param ratio_IF_max: maximum IF ratio at one time when premium is lower than prm_min_IF
        :param expense_fixed: fixed exoense
        :param expense_variable: variable expense
        :param interest: interest rate as 1% = 0.01
        :param initial_cash:
        :param mult_action: multiple applied to action input to this environment. aimed for discrete agent.
        '''

        self.n_agents = n_agents
        self.term = term
        self.total_volume = total_volume
        self.prm_min_NB, self.prm_max_NB, self.ratio_NB_max = prm_min_NB, prm_max_NB, ratio_NB_max
        self.prm_min_IF, self.prm_max_IF, self.ratio_IF_max = prm_min_IF, prm_max_IF, ratio_IF_max
        self.expense_fixed = expense_fixed
        self.expense_variable = expense_variable
        self.interest = interest
        self.initial_cash = initial_cash
        self.mult_action = mult_action

        self.reset()


    def step(self, action, auto_reset=True, verbose=False):
        # action is premium, shape = n_agents, 1
        #
        if np.all(self.done):
            return self.state(), \
                   np.zeros_like(self.done, dtype=float).reshape(-1, 1), \
                   np.array(['no_info',] * self.n_agents).reshape(-1, 1),\
                   np.copy(self.done).reshape(-1, 1)

        if verbose:
            state_current = self.state()

        # define state transition
        NB = self.total_volume - self.IF
        IF = self.IF
        ratio_NB = np.clip(
            1 - 1/ (self.prm_max_NB - self.prm_min_NB) * (self.mult_action * action.ravel() - self.prm_min_NB),
            0, 1) * self.ratio_NB_max
        ratio_IF = np.clip(
            1 - 1 / (self.prm_max_IF - self.prm_min_IF) * (self.mult_action * action.ravel() - self.prm_min_IF),
            0, 1) * self.ratio_IF_max
        self.IF = NB * ratio_NB + IF * ratio_IF  # i.e. sales

        reward = self.cash * self.interest + (
                    self.IF * (self.mult_action * action.ravel() - self.expense_variable) - self.expense_fixed) * (
                             1 + self.interest)
        self.cash = self.cash + reward
        self.t = self.t + 1

        state_next = self.state()  # 終端状態になるとオートリセットになるのでここで一旦取得しておく．

        self.done = np.logical_or(self.cash < 0, self.term == self.t)
        done = np.copy(self.done)
        if auto_reset and np.any(self.done):
            self.t[done] = 0
            self.cash[done] = self.initial_cash
            self.IF[done] = 0
            self.done[:] = False

        if verbose:
            print('S_c = {},  A = {}, R = {}, S_n = {}, done = {}' \
                  .format(state_current, action, reward, state_next, done))

        return state_next, reward.reshape(-1, 1), np.array(['no_info',] * self.n_agents).reshape(-1, 1), done.reshape(-1, 1)


    def reset(self):
        self.t = np.zeros(shape=(self.n_agents,))  # observable
        self.cash = np.ones(shape=(self.n_agents,)) * self.initial_cash  # observable
        self.IF = np.zeros(shape=(self.n_agents,))  # observable
        self.done = np.zeros(shape=(self.n_agents,), dtype=bool)


    def state(self):
        '''
        状態情報を返す．尚，時刻tは必ず含まれていなければならない．
        現実の経済環境では終わりがないので不要だが，シミュレーション上は最終時点に近いかどうかで最適行動が異なる．
        具体的には，初期の場合は保有を稼ぐために低価格設定することが最適だが，
        終了間近では後先考えずに高価格にすることが最適である場合がある（設定に依存する）．
        このとき，状態情報にtが含まれていないと，同じ状態情報に対して複数の最適行動が存在することになるため，学習が収束しない．
        :return: obserbable for agent is [t, cash, IF].
        '''
        return np.copy(self.t.reshape(-1, 1)), np.copy(self.cash.reshape(-1, 1)), np.copy(self.IF.reshape(-1, 1))


    def shapes(self):
        return [S.shape[1:] for S in self.state()]


    def get_config(self):
        config = {}
        config['n_agents'] = self.n_agents
        config['term'] = self.term
        config['total_volume'] = self.total_volume
        config['prm_min_NB'] = self.prm_min_NB;
        config['prm_max_NB'] = self.prm_max_NB;
        config['ratio_NB_max'] = self.ratio_NB_max
        config['prm_min_IF'] = self.prm_min_IF;
        config['prm_max_IF'] = self.prm_max_IF;
        config['ratio_IF_max'] = self.ratio_IF_max
        config['expense_fixed'] = self.expense_fixed
        config['expense_variable'] = self.expense_variable
        config['interest'] = self.interest
        config['initial_cash'] = self.initial_cash

        return config


    ####################################################################################
    # below code are for visualization of setting
    def show_demand(self, figsize=(15, 5), dpi=150, fontsize=16):
        # premium disttibution
        prm_min = min(self.prm_min_NB, self.prm_min_IF)
        prm_max = max(self.prm_max_NB, self.prm_max_IF)
        action = np.linspace(prm_min, prm_max, 101)

        # demand
        ratio_NB = np.clip(
            1 - 1/ (self.prm_max_NB - self.prm_min_NB) * (action.ravel() - self.prm_min_NB),
            0, 1) * self.ratio_NB_max
        ratio_IF = np.clip(
            1 - 1 / (self.prm_max_IF - self.prm_min_IF) * (action.ravel() - self.prm_min_IF),
            0, 1) * self.ratio_IF_max

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(ratio_NB, label='NB')
        ax1.plot(ratio_IF, label='IF')
        ax1.set_title('demand curve', size=fontsize)
        ax1.set_xlabel('premium', size=fontsize)
        ax1.set_ylabel('demand ratio for volume', size=fontsize)
        ax1.legend(fontsize=fontsize)

        # imidiate reward
        IF = np.linspace(0, 1, 101).reshape(-1, 1)
        NB = 1 - IF
        IF = NB * ratio_NB.reshape(1, -1) + IF * ratio_IF.reshape(1, -1)  # i.e. sales
        reward = (IF * (action.reshape(1, -1) - self.expense_variable) - self.expense_fixed)

        ax2 = fig.add_subplot(1, 2, 2)
        sns.heatmap(reward, square=True, ax=ax2, cmap='jet')
        ax2.set_title('imidiate reward', size=fontsize)
        ax2.set_xlabel('action (premium)', size=fontsize)
        ax2.set_ylabel('IF at beginning of year', size=fontsize)

        plt.tight_layout()
        plt.show()


    ####################################################################################
    def show_simulation(self, n_agents=100000,
                        figsize1=None, figsize2=(15, 5), figsize3=(15, 5),
                        dpi1=150, dpi2=150, dpi3=150,
                        fontsize2=16, fontsize3=16):
        n_agents_current = self.n_agents
        mult_action_current = self.mult_action
        self.n_agents = n_agents
        self.mult_action = 1.0
        self.reset()

        done_current = np.zeros(shape=(self.n_agents, 1))
        A = []
        D = []

        if figsize1 is None:
            figsize1 = (15, 3 * (self.term // min(self.term, 5) + 1))

        fig = plt.figure(figsize=figsize1, dpi=dpi1)
        for t in range(self.term):
            state = self.state()
            ax = fig.add_subplot(self.term // min(self.term, 5) + 1, min(self.term, 5), 1 + t)
            ax.scatter(state[1][np.logical_not(done_current)], state[2][np.logical_not(done_current)],
                       c='r', s=0.1, alpha=0.1)  # cash, IF
            ax.set_title(t)
            # ax.set_xlabel('cash'); plt.set_ylabel('IF')
            a = self.prm_min_NB + np.random.uniform(size=(self.n_agents, 1)) * (self.prm_max_IF - self.prm_min_NB)
            A.append(a)
            _, _, _, done = self.step(a, auto_reset=False)
            done_current = np.logical_or(done_current, done)
            D.append(np.copy(done_current))

        plt.tight_layout()
        plt.show()

        # show best actions and distribution of score
        A = np.array(A)  # A.shape = T, n_agents, 1
        D = np.array(D)
        max_R = np.max(self.state()[1])
        min_R = np.min(self.state()[1])
        best_a = A[:, np.argmax(self.state()[1].ravel()), 0]
        fig = plt.figure(figsize=figsize2, dpi=dpi2)

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(best_a, lw=3.0, label='best')

        for i in range(0, A.shape[1], 100):
            ax1.plot(A[np.concatenate([np.array([True, ]), ~D[:-1, i, 0]]), i, 0],
                     alpha=((self.state()[1].ravel()[i] - min_R) / (max_R - min_R)) ** 6, c='r',
                     label='others' if i == 0 else None)
        ax1.set_title('best actions', size=fontsize2)
        ax1.set_xlabel('t', size=fontsize2)
        ax1.set_ylabel('action (premium)', size=fontsize2)
        ax1.legend(fontsize=fontsize2)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.hist(self.cash, bins=20)
        ax2.set_title('histogram of return', size=fontsize2)
        ax2.set_xlabel('return', size=fontsize2)

        plt.tight_layout()
        plt.show()

        # show survival plot
        plt.figure(figsize=figsize3, dpi=dpi3)
        sns.heatmap(1 - D[:, ::1, 0].T)
        plt.title('survival plot', size=fontsize3)
        plt.show()

        R = self.state()[1].ravel()
        print('max(R) = {:.3f}, min(R) = {:.3f}, mean(R) = {:.3f}, std(R) = {:.3f}'
              .format(np.max(R), np.min(R), np.mean(R), np.std(R)))
        print('survival ratio at end = {:.2%}'.format(1 - np.mean(D[-2])))


        # environment recover
        self.n_agents = n_agents_current
        self.mult_action = mult_action_current
        self.reset()


    ####################################################################################


########################################################################################################################
if __name__ == '__main__':
    env = YRTRenewal()
