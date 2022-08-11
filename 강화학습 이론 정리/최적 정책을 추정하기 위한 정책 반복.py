"""
<pi~ pi_* 의 추정을 위한 반복 정책 평가>

<설정해야 할 것>
epsilon : 상수
v = V(s)
S

def V(s) :
def r(s',a,s) :
def pi(s,a) : s가 주어졌을 때 a를 선택할 확률
def action_pi(s) : S -> A : s가 주어졌을 때 (결정론적) 행동
def p(s',r,s,a) : 결정론적 이동일 때에는 1로 고정. 확률적 이동일때 값을 정의해야함.

argmax_a q_*

"""

import random
import numpy as np
import math

class Policy_Evaluation:
    def __init__(self, s, a, gamma=0.9, epsilon=0.0000001, det=True):
        self.epsilon = epsilon
        self.gamma = gamma
        self.det = det
        self.S = s# 상태 항목들 list
        self.A = a# 행동 항목들 list
        self.v_list, self.a_list = self.initiate()
        self.count = 100

    def initiate(self):
        v_list = []
        a_list = []
        for s in self.S :
            v_list.append(self.V(s))
            a_list.append(self.action_pi(s))
        return v_list, a_list

    def r(self, next_s, a, s):
        pass

    def pi(self, s, a):
        pass

    def action_pi(self, s):
        if self.det == True :
            pass
            # 결정론적이면 값이 하나만 나옴
        pass
        # 확률적이면 값이 list 형태로 나옴

    def next_s(self, s, A): # 상태 s에서 a 행동을 했을 때 이동하는 곳. 여러 가능성을 list 형태로 반환
        pass
        if type(A) == list : # A가 list로 주어지면 이중 리스트로 표현
            ans = []
            for a in A :
                ans.append(self.next_s(s,a))
            return ans

    def p(self, next_s, r, s, a, det=self.det): # a부분에 list를 넣으면 list의 형태로 값이 나오도록 수정 필요.
        if det == True : return 1
        else : pass

    def V(self, s): #상태 s를 입력했을 때의 가치 함수의 단일 값
        sum_all = 0
        if self.det == True :
            for next_s in self.next_s(s,self.action_pi(s)) :
                r = self.r(next_s, self.action_pi(s),s)
                sum_all += self.p(next_s, r, s, self.action_pi(s))*(r+ self.gamma*self.V(next_s))
            return sum_all

        for a in self.action_pi(self, s): #actino_pi 의 a list
            for next_s in self.next_s(s, a): # 각 a 별 next_s list
                r= self.r(next_s, a, s)
                sum_all += self.pi(s,a)*self.p(next_s, r, s, a)*(r+ self.gamma*self.V(next_s))
            return sum_all


    def state_evaluate(self):
        next_v_list = self.v_list
        n = 0
        while delta < self.epsilon:
            n += 1
            delta = 0
            v_list = next_v_list
            for index, s in enumerate(self.S):
                next_v_list[index] = self.V(s)
            diff = math.sqrt(sum([(v_list[index] - next_v_list[index])**2 for index in range(len(v_list))]))
            delta = max(delta, diff)
        self.count = n

        self.v_list = next_v_list
            # v의 값이 수렴해 있겠네. 그런데 지금의 형태는 s가 바뀔 때마다 V(s)의 값이 바뀌는 구조 아닌가?
            # 각 s별로 V(s)값이 따로 저장되어 있어야 겠는데? list의 형태로. (조치완료)
            # (조치완료) 계산 한 값은 self.v_list에 저장하여 다른 곳에서도 사용할 수 있도록 설정

    def q(self, s, a):
        sum_all = 0
        for next_s in self.next_s(s, a):
            r= self.r(next_s, a, s)
#            sum_all += self.p(next_s, r, s, a)*[self.r(next_s,a,s) + self.gamma*self.V(next_s)]
            sum_all += self.p(next_s, r, s, a) * [self.r(next_s, a, s) + self.gamma * self.v_list[next_s]]
        # 계산을 반복적으로 하지 않기 위해서 v_list 에서 next_t의 값을 불러옴.
        return sum_all

    def improve_policy(self):
        safety = True
        action = self.a_list # 수정 필요
        for s in self.S:
            before_action = action
            lst, a_lst = [], []

            for a in self.action_pi(s) : # action_pi(s) 가 단일 값을 반환한다면 오류가 생길듯 ㅠ
                lst.append(self.q(s, a))
                a_lst.append(a)
            action[s] = a_lst[np.argmax(lst)]
            # 일부 수도 코드 수정. 기존 상태라면 매번 이전행동 != action[s]를 비교해야함.
        if before_action != action :
            safety = False

        if safety == True or self.count == 1:
            return self.v_list, self.a_list

        self.state_evaluate()


# v 값을 어떻게 비교할 수 있을까? self.count 도입 &