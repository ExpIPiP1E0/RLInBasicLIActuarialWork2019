import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=3)  # 数値桁数を指定．


########################################################################################################################
class Environment(object):
    '''
    責任準備金環境：
    若干混乱しやすいのだが，年始で一切のCFの発生前が各状態となる（即ち生保数理でいうところのt基準）．
    責任準備金としては，どれだけの資産を残すかを毎年年初に決定する．
    年末にCFが生じて支払不能の場合は破産ペナルティを生じてゲームエンドとする．
    また，CFが最終地点まで到達してもゲームエンドとする．
    即時報酬は年始に放出する配当及びゲームエンド時の残存資産に1年分のハードルレートを適用したモノとする．
    内部状態としては，実質的には時間t・年始時点キャッシュcashの2つのみである．
    DNNに渡す段階では(n_agents, state_dims)の形態である必要があるが，内部的にはstate_dimsは不要なため，ベクトルとして持つ．
    '''

    def __init__(self, cashflow, interest=0.01, hurdle=0.00,
                 initial_cash=10, bankrupt_penalty=-1000,
                 bankrupt_info=True,
                 n_agents=64):
        self.cashflow = cashflow
        self.interest = interest
        self.hurdle = hurdle
        self.initial_cash = initial_cash
        self.bankrupt_penalty = bankrupt_penalty
        self.bankrupt_info = bankrupt_info
        self.n_agents = n_agents

        self.done_counter = 0
        self.done_bankrupt_counter = 0

        self.reset()


    def step(self, action, auto_reset=True, verbose=False):
        # 連続型対応
        action = np.array(np.round(action), dtype=int).ravel()

        penalty_BoY = np.zeros(shape=(self.n_agents))  # 年始ペナルティ
        penalty_EoY = np.zeros(shape=(self.n_agents))  # 年末ペナルティ

        # 配当決定及びそれによる破産．
        action = np.minimum(action, self.cash)  # 問題を簡易化するために，配当をキャップする．
        dividend = self.cash - action  # actionは責任準備金の額とする．
        self.done[dividend < 0] = True  # 負配当はルール違反で破産．但し，これは実際には機能しないようにしている．
        penalty_BoY[self.done] = self.bankrupt_penalty  # 年始ペナルティ（過剰配当）

        # 1年経過
        self.cash = action * (1 + self.interest) + self.cashflow[self.t]
        self.t = self.t + 1

        # 終端状態判定
        self.done[self.t == len(self.cashflow)] = True  # 時間的な終わり
        self.done[self.cash < 0] = True  # 破産
        penalty_EoY[self.cash < 0] = self.bankrupt_penalty  # 年末ペナルティ（破産）
        info = np.where(self.cash < 0, 'bankrupt' if self.bankrupt_info else 'ok', 'ok').reshape(-1, 1)
        self.done_bankrupt_counter += np.sum(self.cash < 0)  # 破産件数カウント
        self.done_counter += np.sum(self.done)

        # 即時報酬〜年始配当+ペナルティ+終了時残存資産
        reward = dividend + penalty_BoY \
                 + (self.cash * self.done + penalty_EoY) / (1 + self.hurdle)

        # 返り値確定
        state_next = self.state()
        done_next = np.copy(self.done)

        # 終端したエージェントをリセット
        if auto_reset and np.any(self.done):
            self.t[self.done] = 0
            self.cash[self.done] = self.initial_cash
            self.done[self.done] = False

        # 終了
        return state_next, reward.reshape(-1, 1), info, done_next.reshape(-1, 1)


    def reset(self):  # 内部的には実質，t,cashの2つのみが状態に相当する．他は全てこれらから導出できる．
        self.t = np.zeros(shape=self.n_agents, dtype=int)
        self.cash = self.initial_cash * np.ones(shape=self.n_agents)
        self.done = np.zeros(shape=self.n_agents, dtype=bool)


    def state(self):  # 外部とやりとりするのは，現在cash, 将来CFのみである．
        future_CF = np.zeros(shape=(self.n_agents, self.cashflow.shape[0]))
        for a in range(self.n_agents):
            future_CF[a, :len(self.cashflow) - self.t[a]] = self.cashflow[self.t[a]:]
        return np.copy(self.cash.reshape(-1, 1)), future_CF


    def shapes(self):
        return [S.shape[1:] for S in self.state()]

