"""
仕様
・random_matrixは今のところ対称行列のみを生じており、そうでない場合はまだ不十分.
・random_matrixに逆行列を持たないものは調べられない.
・対角成分でないところがMPMaximizerだった場合はまだ不正確.
・payoff_matrixは次元が増えても対応.ただし、利得が対称なゲームに限る.
・それぞれの戦略の組に対して、random_matrixは最大で100回発生させる.
・計算にかかる時間は手元のノートPC上では２次元で１秒以内、３次元で約２秒、４次元で約18秒.
"""

import numpy as np
from random import uniform
from __future__ import division
from math import floor
from itertools import combinations

class MPMaximizer:
    def __init__(self, mat):
        self.payoff_matrix = mat
        self.dim = mat.shape[0]
        self.strategies = self.gen_strategies()
        self.trial = 100
        
    def find(self, strict):
        """
        対角成分でないところがMPMaximizerだった場合は「空振る」恐れがある.
        発生する行列は対称な行列なので、対角成分に関しては「空振り」はない.
        いずれにしても、trialを無限に大きくしない限り、「見逃し」の可能性は消えない.
        """
        for i in range(len(self.strategies)):
            print "a-star = " + str(self.strategies[i])

            self.strategy = self.strategies[i][0]
            self.setstrategy = self.strategies[i]
            
            self.degU_payoff_matrix = self.degen_matrix(self.payoff_matrix, True)
            self.degL_payoff_matrix = self.degen_matrix(self.payoff_matrix, False)            
            
            for j in range(self.trial):        
                self.random_matrix = self.gen_random_matrix()
        
                self.degU_random_matrix = self.degen_matrix(self.random_matrix, True)
                self.degL_random_matrix = self.degen_matrix(self.random_matrix, False)
        
                self.equilibria = self.gen_equilibria()
                
                self.br1st_random = self.cal_br(self.degU_random_matrix, True)
                self.br2nd_random = self.cal_br(self.degL_random_matrix, False)
                
                if strict:
                    self.br1st_payoff = self.cal_br(self.degU_payoff_matrix, False)
                    self.br2nd_payoff = self.cal_br(self.degL_payoff_matrix, True)
                else:
                    self.br1st_payoff = self.cal_br(self.degU_payoff_matrix, True)
                    self.br2nd_payoff = self.cal_br(self.degL_payoff_matrix, False)
        
                self.cond1st = self.compare(self.br1st_random, self.br1st_payoff)
                self.cond2nd = self.compare(self.br2nd_payoff, self.br2nd_random)
                
                if self.cond1st and self.cond2nd:
                    print self.random_matrix
                    break
            
            if j == self.trial - 1:
                print "This a-star does not seem to be MP-maximizer."
        
    def set_trial(self, trial):
        self.trial = trial
        
    def gen_equilibria(self):
        """
        ２つ以上の戦略が無差別になる空間と、dim次元の(0,1)空間の境界部分との共通点を全て挙げる.
        (例)dim = 3ならば、まず３つの戦略が無差別になる点を探す.
            次に、２つの戦略が無差別になる直線が３本（(0,1),(0,2),(1,2)それぞれ）あり、
            これとの３次元空間の(0,1)との境界部分は多くて直線１本につき２ヶ所だから６点が加わり、
            高々７点がequilibriaに加えられる.
        """
        equilibria = []
        
        for i in range(self.dim - 1):
            rawcolumns = self.gen_deletelist(i)
            for j1 in range(len(rawcolumns)):
                for j2 in range(len(rawcolumns)):
                    self.cut_matrix(rawcolumns[j1], rawcolumns[j2])
                    equilibrium = self.cal_equilibrium()
                    
                    if self.test_area(equilibrium):
                        for k in rawcolumns[j2]:
                            equilibrium = np.insert(equilibrium, k, 0)
                        equilibria.append(equilibrium)
        
        for i in range(self.dim):
            zeros = np.zeros(self.dim - 1)
            zeros = np.insert(zeros, i, 1)
            equilibria.append(zeros)
        
        return equilibria
                    
    def test_area(self, vec, epsilon=1E-7):
        """
        与えられた混合戦略が確率分布ならばTrue、そうでなければFalseを返す.
        すなわち、全ての戦略の割合が非負ならばTrue.
        """
        boolvec = vec > -epsilon
        return self.cut_dim - np.count_nonzero(boolvec) == 0       
    
    def get_dim(self):
        return self.dim
    
    def gen_deletelist(self, dim):
        """
        equilibriaを求めるのに必要.
        (例)combinations(range(6), 4)は、range(6)=(0, 1, 2, 3, 4, 5)から4つを選び順番に並べる組を全て返す.
            すなわちこの場合、deletelistに加えられるのは
            (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 5), (0, 1, 3, 4), (0, 1, 3, 5), (0, 1, 4, 5),
            (0, 2, 3, 4), (0, 2, 3, 5), (0, 2, 4, 5), (0, 3, 4, 5),
            (1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 4, 5), (1, 3, 4, 5), (2, 3, 4, 5)の15個.
        """
        deletelist = []
        for i in combinations(range(self.dim), dim):
            deletelist.append(i)
        return deletelist
    
    def cut_matrix(self, raws, columns):
        """
        equilibriaを求めるのに必要.
        引数の「行」「列」を取り除く.
        """
        self.cutmatrix = self.random_matrix
        self.cutmatrix = np.delete(self.cutmatrix, raws, 0)
        self.cutmatrix = np.delete(self.cutmatrix, columns, 1)
        self.cut_dim = self.cutmatrix.shape[0]
    
    def gen_random_matrix(self):
        """
        発生させているのは対称行列のrandom_matrix.
        """
        random_matrix = np.random.uniform(0, 1, (self.dim, self.dim))
        random_matrix = (random_matrix + random_matrix.T) * 0.5
        random_matrix[self.setstrategy] = 1
        return random_matrix
    
    def cal_equilibrium(self):
        """
        与えられた利得行列に対し、BRが全ての戦略になるような相手の混合戦略（負の値を認める）を求める.
        条件は、利得行列が逆行列を持つこと.
        式で書くと、
        M = 与えられた行列（dim*dim次元の行列）
        x = 求める混合戦略（dim次元ベクトル）
        1 = 全ての成分が1のdim次元ベクトル
        c = 実数定数
        としたときに、
        Mx = c1 and x'1 = 1
        を満たすcとxを求め、xを返す.
        """
        ones = np.ones(self.cut_dim)
        inversemat = np.linalg.inv(self.cutmatrix)
        x = np.dot(inversemat, ones)
        c = np.dot(x.T, ones)
        return x / c
    
    def gen_strategies(self):
        """
        次元を与えると、取りうる全ての純粋戦略の組を返す.
        playerが２人の場合にのみ対応.
        (例)dim = 2ならば、(0, 0), (0, 1), (1, 0), (1, 1)の4つをリスト中のタプルとして返す.
        """
        strategies = []
        for i in range(self.dim):
            for j in range(self.dim):
                strategies.append((i,j))
        return strategies

    def degen_matrix(self, mat, upper, HUGE=1E+7):
        """
        行列を退化させる関数.BestResponseを求めるが、MPMaximizerの戦略a-starによって、
        条件式で取れる戦略を限る必要があるので、この関数が必要.
        命名はupperがTrueなら下の例での１つ目、Falseなら２つ目になるよう返す.
        (例)A_i = {0, 1, 2, 3}でa-star_i = 2とすると、２つある条件のうちの１つ目
        br{pi_i|[0,2]}は、0から2までの戦略の中からbrを選ぶということなので、
        利得行列を、
        [ ## ## ## ##
        ## ## ## ##
        ## ## ## ##
        -- -- -- -- ]
        （ただし##は元の値、--は##に比べて明らかに小さい値（-- << ##））
        とすることで3という戦略が他に支配されるようにできる.同様に、２つ目の条件
        br{pi_i|[2,3]}は、利得行列を
        [ -- -- -- --
        -- -- -- --
        ## ## ## ##
        ## ## ## ## ]
        とする.
        """
        degen = HUGE * np.ones((self.dim, self.dim))
        if upper:
            for i in range(self.strategy+1):
                degen[i, :] = 0
        else:
            for i in range(self.strategy, self.dim):
                degen[i, :] = 0
        return mat - degen

    def cal_br(self, deg_mat, maximum, epsilon=10E-7):
        """
        それぞれの混合戦略に対応した順番にBestResponseを返す関数.
        BestResponseが複数ある場合はmaximumがTrueなら最大値を、そうでなければ最小値を返す.
        """
        best_responses = []
        for i in range(len(self.equilibria)):
            A = np.dot(deg_mat, self.equilibria[i])
            br = 0
            value = A[0]
            for now_number in range(self.dim):
                if maximum:
                    if A[now_number] - value > -epsilon:
                        br = now_number
                        value = A[now_number]
                else:
                    if A[now_number] - value > epsilon:
                        br = now_number
                        value = A[now_number]
            best_responses.append(br)
        return best_responses    
        
        A = np.dot(deg_mat, self.equilibria)
        number_of_colomns = len(test[0])
        best_responses = []
        for i in range(number_of_colomns):
            br = 0
            value = A[0][i]
            for now_number in range(dim):
                if maximum:
                    if A[now_number][i] - value > -epsilon:
                        br = now_number
                        value = A[now_number][i]
                else:
                    if A[now_number][i] - value > epsilon:
                        br = now_number
                        value = A[now_number][i]
            best_responses.append(br)
        return best_responses

    def compare(self, smaller, bigger):
        """
        リスト内の対応する要素の大小を比べる.全ての要素でsmaller側がbigger側よりも「<=」ならばTrue、
        １ヶ所でもそうなっていなければFalseを返す.
        """
        X = []
        for i in range(len(smaller)):
            X.append(smaller[i] <= bigger[i])
        return all(X)
    
    def get_1st(self):
        print self.br1st_payoff
        print self.br1st_random
    
    def get_2nd(self):
        print self.br2nd_payoff
        print self.br2nd_random
        
    def get_equilibria(self):
        print self.equilibria