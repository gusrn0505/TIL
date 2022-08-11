"""
<V ~ v_pi 의 추정을 위한 반복 정책 평가>

평가받을 정책 pi가 입력으로 들어간다.
알고리즘의 파라미터는 추정의 정확도를 결정한느 작은 기준값 epsilon >0이다.
V(s)를 모든 s in S 에 대해 임의의 값으로 초기화한다.
다만 V(종단) = 0으로 한다.

<설정해야 할 것>
epsilon : 상수

delta : max(delta, |v-V(s)|)
v = V(s)
S

def V(s) :
def r(s',a,s) :
def pi(s,a) :
def p(s',r,s,a) : 결정론적 이동일 때에는 1로 고정. 확률적 이동일때 값을 정의해야함.
"""
import random

class Evaluation :
    def __init__(self, S, A, gamma = 0.9, epsilon = 0.0000001, det=True):
        self.epsilon = epsilon
        self.gamma = gamma
        self.det = det
        self.S = S # 상태 항목들 list
        self.A = A # 행동 항목들 list

    def r(self, next_s, a, s):
        pass

    def pi(self, s, a):
        pass

    def next_s(self, s, a): # 상태 s에서 a 행동을 했을 때 이동하는 곳. 여러 가능성을 list 형태로 반환
        pass

    def p(self, next_s, r, s, a, det=self.det):
        if det == True : return 1
        else : pass

    def V(self, s): #상태 s를 입력했을 때의 가치 함수의 값
        sum_all = 0
        for a in self.A :
            for next_s in self.next_s(s,a) :
                r = self.r(next_s, a,s)
                sum_all += self.pi(s,a)*self.p(next_s, r, s, a)*(r+ self.gamma*self.V(next_s))
        return sum_all

    def evaluate(self):
        next_v = self.V(random.choice(self.S))

        while delta < self.epsilon :
            delta = 0

            for s in self.S :
                v = next_v
                next_v = self.V(s)
                delta = max(delta, abs(v-next_v))
