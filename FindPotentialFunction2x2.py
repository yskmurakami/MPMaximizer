import numpy as np
from random import uniform

class SolveEquivalence2x2:
    
    def __init__(self, matrix, dim=2):
        self.x = np.zeros(dim)
        for i in range(dim):
            self.x[i] = matrix[0, i] - matrix[1, i]

    def set_epsilon(self, new_value):
        self.epsilon = new_value
        
    def solve(self, epsilon=10E-7):
        if -epsilon < self.x[0] - self.x[1] < epsilon:
            print "Error : p is too much."
        else:
            return - self.x[1] / (self.x[0] - self.x[1])

class BR2x2:
    
    def __init__(self, matrix, p, delta=10E-3):
        self.u = matrix
        self.p = np.array([[0, p-delta, p, p+delta, 1], np.ones(5)-[0, p-delta, p, p+delta, 1]])

    def slicemat0x(self, x, huge=10E+7):
        if x == 0:
            self.u = self.u - np.array([np.zeros(2), huge * np.ones(2)])
    
    def slicematxn(self, x, huge=10E+7):
        if x == 1:
            self.u = self.u - np.array([huge * np.ones(2), np.zeros(2)])
        
    def solve(self):
        self.br = np.dot(self.u, self.p)
    
    def get_br(self):
        return self.br
    
    def maxbr(self, epsilon=10E-10):
        self.bru = np.zeros(5)
        for i in range(5):
            self.bru[i] = ((int)(self.br[1, i] - self.br[0, i] > - epsilon))
        return self.bru
    
    def minbr(self, epsilon=10E-10):
        self.bru = np.zeros(5)
        for i in range(5):
            self.bru[i] = ((int)(self.br[1, i] - self.br[0, i] > epsilon))
        return self.bru
    
    def get_bru(self):
        return self.bru

def RandomMatrix2x2(x, y):
        v = np.random.uniform(0, 1, size=(2, 2))
        v[x, y] = 1.0
        return v
    
class FindPotentialFunction:
    
    def __init__(self, x, y, mat, p):
        self.x = x
        self.y = y
        self.flag = x == y
        self.mat = mat
        self.p = p
    
    def condition(self):
        q = BR2x2(self.mat, self.p)
        q.slicemat0x(self.x)
        q.solve()
        self.bru1 = q.maxbr()
        
        q = BR2x2(self.mat, self.p)
        q.slicematxn(self.x)
        q.solve()
        self.bru2 = q.minbr()

        if not self.flag:
            q = BR2x2(self.mat, self.p)
            q.slicemat0x(self.y)
            q.solve()
            self.bru3 = q.maxbr()
            
            q = BR2x2(self.mat, self.p)
            q.slicematxn(self.y)
            q.solve()
            self.bru4 = q.minbr()
    
    def find(self, iteration=100):
        i = 0
        while i < iteration:
            poten = RandomMatrix2x2(self.x, self.y)
            
            q = BR2x2(poten, self.p)
            q.slicemat0x(self.x)
            q.solve()
            self.brv1 = q.minbr()
            
            q = BR2x2(poten, self.p)
            q.slicematxn(self.x)
            q.solve()
            self.brv2 = q.maxbr()
            
            if not self.flag:
                q = BR2x2(poten, self.p)
                q.slicemat0x(self.y)
                q.solve()
                self.brv3 = q.minbr()
                
                q = BR2x2(poten, self.p)
                q.slicematxn(self.y)
                q.solve()
                self.brv4 = q.maxbr()
                
                if (self.brv1 - self.bru1).max() <= 0 and (self.brv2 - self.bru2).min() >= 0 and (self.brv3 - self.bru3).max() <= 0 and (self.brv4 - self.bru4).min() >= 0:
                    return poten
                else:
                    i += 1
            else:
                if (self.brv1 - self.bru1).max() <= 0 and (self.brv2 - self.bru2).min() >= 0:
                    return poten
                else:
                    i += 1
                
        if i == 100:
            print "Error : there seems to be no potential maximizer."
        
