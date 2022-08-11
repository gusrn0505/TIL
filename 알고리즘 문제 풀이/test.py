import random
import numpy as np
import math



class Policy_Evaluation:
    def __init__(self, s, a, gamma=0.9, epsilon=0.0000001, det=False):
        self.epsilon = epsilon
        self.gamma = gamma
        self.det = det
        self.S = s# 상태 항목들 list
        self.A = a# 행동 항목들 list
        self.S_A = self.initiate_S_A()
        self.v_list, self.a_list = self.initiate()
        self.count = 100
        self.A_rent_avg = 3
        self.A_return_avg = 4
        self.B_rent_avg = 3
        self.B_return_avg = 2

    def initiate(self):
        v_list = []
        a_list = []
        for s in self.S :
            v_list.append(self.V(s))
            a_list.append(self.action_pi(s)) # action_pi 정의에 따라 달라질 수 있음.
        return v_list, a_list

    def initiate_S_A(self):
        dic = {}
        # dict 에 list는 key로 넣을 수 없어 tuple() 사용함.
        for s in self.S :
            dic[tuple(s)] = [0,0,0,0,0,1,0,0,0,0,0]
        return dic


    def prob_foisson(self,x, lamda):
        return lamda ** x * math.e**(-lamda) / math.factorial(x)

    def r(self, next_s, a, s):
        pass

    def alt_r(self, A, B, a_rent, b_rent, a): #A,B는 각각 A,B 지점의 상태를 의미
        actual_rent = [min(A[0], a_rent), min(B[0], b_rent)]
        actual_a = min((A[0] - actual_rent[0]) - (B[0] - actual_rent[1]), a)
        return sum(actual_rent)*10 - 2*actual_a

    def pi(self, s, a): # 단일 확률 값으로 계산되어야 함. dic에서 값 불러오기
        return self.S_A[tuple(s)][a+5]

    def action_pi(self, s):
        action_lst = [index-5 for index, prob in enumerate(self.S_A[tuple(s)]) if prob > 0]
        return action_lst

    def next_s(self, s, A): # 상태 s에서 a 행동을 했을 때 이동하는 곳. 여러 가능성을 list 형태로 반환
        # A가 list로 주어지면 단일값으로 쪼개기. 그리고 결과값을 이중 리스트로 표현
        if type(A) == list :
            ans = []
            for a in A :
                ans.append(self.next_s(s,a))
            return ans

        # 단일 행동이 부여되었을 때 다음 결과값 부여
        # 입력 변수를 줄이기 위해서, s의 값은 이미 각각 대여를 끝낸 이후여야 한다.
        if A > 0 :
            actual_a = min(s[0], A)
            return [s[0] - actual_a, s[1] + actual_a]
        else :
            actual_a = min(s[1], -A) #
            return [s[0] + actual_a, s[1] - actual_a]


S = sum([[[i,j] for i in range(21)] for j in range(21)], [])
A = list(range(-5,6))

test = Policy_Evaluation(S,A)
print(test.prob_foisson(3,4))
print(test.alt_r([1,2], [2,3], 2,4,1))
print(test.action_pi([2,3]))
print(test.pi([1,2], 0))
print(test.next_s([3,2], [-3, 2, 4]))

print(test.S_A[(0,0)])