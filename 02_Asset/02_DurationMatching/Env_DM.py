import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=3)  # 数値桁数を指定．


########################################################################################################################
class Environment(object):
    def __init__(self, yield_curve, liability_CF, n_agents=64,
                 gen_yield_on_reset=False, yield_start=0.00, yield_end=0.10, yield_vol=0.0,
                 reset_cash=True):  # verified.
        '''

        :param yield_curve: yield curve shape = (tenor=1+T, T), i.e. projection of each tenor. 1+T's 1 means cash holding.
        :param liability_CF: array of liability CF shape = (T, ). treated as EoY CF.
        :param n_agents:
        :param gen_yield_on_reset: specify reset yield when reset. if so, define below parameters
        :param yield_start:
        :param yield_end:
        :param yield_vol:
        :param reset_cash: reset cash to reserve at very year end
        '''
        self.yield_curve = np.tile(np.expand_dims(yield_curve, axis=0), reps=(n_agents, 1, 1))  # for n_agents, tenor, maturity_term(for t)
        self.liability_CF = np.tile(np.array(liability_CF).reshape(1, -1), reps=(n_agents, 1))  # n_agents, maturity_term
        self.n_agents = n_agents
        self.maturity_term = len(liability_CF)
        self.reset_cash = reset_cash

        self.gen_yield_on_reset = gen_yield_on_reset
        if self.gen_yield_on_reset:
            self.yield_start = yield_start
            self.yield_end = yield_end
            self.yield_vol = yield_vol

        self.reset()


    def step(self, action, auto_reset=True, verbose=False):
        # 今のところ満期時ペイアウトのタイプにしか対応していない．
        info = np.full(shape=(self.n_agents, 1), fill_value='no_info')
        action = np.clip(np.round(action), 0, self.maturity_term).astype(int)

        # 既に終端状態の場合は何もせずに現状維持．
        if self.t == self.maturity_term:
            return self.state(), \
                   np.zeros(shape=(self.n_agents, 1)), \
                   info, \
                   np.ones(shape=(self.n_agents, 1), dtype=bool)

        # 年始状態
        cash_BoY = self.cash.ravel()  # 年始資産価格（市場価値ベース）
        reserve_BoY = -self._PV_liability_CF(self.t, BoY=True)  # 年始責任準備金（市場価値ベース）
        net_asset_BoY = cash_BoY - reserve_BoY  # 年始資産負債差額（市場価値ベース）

        # 資産満期価格
        maturity_value = cash_BoY * \
                         ((1 + self.yield_curve[np.arange(self.n_agents), action.ravel(), self.t]) \
                          ** (action.ravel()))

        self.t += 1
        cash_EoY = maturity_value / ((1 + self.yield_curve[np.arange(self.n_agents),
                                                           np.maximum(0, action.ravel() - 1),
                                                           min(self.t, self.maturity_term - 1)])
                                     ** np.maximum(0, action.ravel() - 1))  # 年末残存期間
        reserve_EoY = -self._PV_liability_CF(self.t-1, BoY=False)  # n_agents,

        if self.t == self.maturity_term:  # 満期時〜yieldは直前と同じとする(self.t-1としている部分)．これは処理の簡易化の為．
            self.done.fill(True)  # n_agents, 1

        if self.reset_cash:
            self.cash = reserve_EoY.reshape(-1, 1)
        else:
            self.cash = cash_EoY.reshape(-1, 1)

        net_asset_EoY = cash_EoY - reserve_EoY  #
        reward = -np.abs(net_asset_EoY - net_asset_BoY).reshape(-1, 1)  # n_agents, 1

        if verbose:
            print('t_BoY = {}, action = {}, cash_BoY = {}, reserve_BoY = {}, cash_EoY = {}, reserve_EoY = {}'.format(
                self.t - 1, action, cash_BoY, reserve_BoY, cash_EoY, reserve_EoY))

        state_next = self.state()  # 終端状態に達していた場合にオートリセットするため，ここで一旦取得しておく．

        done = np.copy(self.done)
        if auto_reset and np.any(self.done):  # 本環境は必ず同時に全てのエージェントが終端状態に達する．
            self.reset()

        return state_next, reward, info, done  # S', R, info, done


    def reset(self):  # verified.
        self.t = 0  # 破産などのイレギュラーな脱退が無いため，tは全エージェントで共通である．

        if self.gen_yield_on_reset:
            self.gen_yield_curve(self.yield_start, self.yield_end, self.yield_vol)  # n_agents, tenor, maturity_term(=t)
        self.cash = -np.ones(shape=(self.n_agents, 1)) * self._PV_liability_CF(t=0).reshape(-1, 1)  # n_agents, 1
        self.done = np.zeros(shape=(self.n_agents, 1), dtype=bool)  # n_agents, 1
        return self.state()


    def state(self):  # verified.
        # 外部出力用．コピーを行うので，クラス内部では用いないこと．
        if self.t < self.maturity_term:  # 満期前（通常時）
            cash = np.copy(self.cash.reshape(-1, 1))  # n_agents, 1
            yield_curve = np.copy(self.yield_curve[:, :, self.t])  # n_agents, tenor, t=t
            liability_CF = np.zeros(shape=self.liability_CF.shape)  # n_agents, maturity_term
            liability_CF[:, : self.maturity_term - self.t] = self.liability_CF[:, self.t:]  # n_agents * (maturity_term - t)
        else:  # 満期
            cash = np.zeros(shape=self.cash.shape)  # n_agents, 1
            yield_curve = np.zeros(shape=self.yield_curve[:, :, 0].shape)  # n_agents, tenor, t=1
            liability_CF = np.zeros(shape=self.liability_CF.shape)  # n_agents, maturity_term

        return cash, yield_curve, liability_CF  # n_agents, (1, tenor, maturity_term)


    def _PV_liability_CF(self, t, BoY=True):  # BoYは年末CFの支払前である．
        liability_CF = np.concatenate([self.liability_CF[:, t:],
                                       np.zeros(shape=(self.n_agents, self.maturity_term))], axis=1)[:, :self.maturity_term]  # n_agents * term_cf
        yield_curve = self.yield_curve[:, :, t if BoY else min(t+1, self.maturity_term-1)]  # n_agents * term_cf

        # BoYのとき，t=0にはtenor=1が使用される．そうでないときは，t=0にはtenor=0が使用される．
        if BoY:
            discount_factor = 1 / (1 + yield_curve[:, 1:])  # n_agents, T
            discount_factor = discount_factor ** (1 + np.arange(self.maturity_term).reshape(1, -1))  # ** 1, T
        else:
            discount_factor = 1 / (1 + yield_curve[:, :-1])  # n_agents, T
            discount_factor = discount_factor ** np.arange(self.maturity_term).reshape(1, -1)  # ** 1, T

        PV_liability_CF = np.sum(discount_factor * liability_CF, axis=1)

        return PV_liability_CF  # n_agents,


    def gen_yield_curve(self, start=0.0, end=0.05, vol=0.2):  # verified.
        # n_agents * tenor * maturity_term
        self.yield_curve = np.ones(shape=self.yield_curve.shape)  # n_agents, tenor, maturity_term
        self.yield_curve = self.yield_curve * np.linspace(start, end, self.yield_curve.shape[1]).reshape(1, -1, 1)
        self.yield_curve = self.yield_curve * (
                    1 + np.clip(np.random.randn(*self.yield_curve.shape) * vol, -1, np.inf))  # multiply volatility


    def shapes(self):
        return [S.shape[1:] for S in self.state()]


    def _duration(self):
        # 検証用．
        pass


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






