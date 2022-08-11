"""
<자동차 렌탈 예제 정책 평가>

<최초 설정>
action_pi(s) = [0]*len(self.S)
S = sum([[[i,j] for i in range(21)] for j in range(21)], [])
A = list(range(-5,6))
#Q. S를 튜플로 줘야하나, list로 줘야하나. => 값이 변경될 수 있게 list로 부여하기

p(s', r| s,a) = p(A_대여, A_반납, B_대여, B_반납)
p(A_대여) = foisson(A_대여수)

<고려점>
A_대여, A_반납, B_대여, B_반납 에 의해서 return과 s' 가 정해진다.
남아있는 차량의 수에 따라 실제 대여되는 것과 행동 a의 반영은 다르다!

<설정해야하는 것>
def prob_foisson(x) :
"""

import random
import numpy as np
import math



class Policy_Evaluation:
    def __init__(self, s, a, gamma=0.9, epsilon=0.0000001, det=False):
        self.epsilon = epsilon
        self.gamma = gamma
        self.det = det
        self.S = s  # 상태 항목들 list
        self.A = a  # 행동 항목들 list
        self.v_list, self.S_A = self.initiate()
        self.count = 100
        self.A_rent_avg = 3
        self.A_return_avg = 4
        self.B_rent_avg = 3
        self.B_return_avg = 2

    def S_A_initiate(self):
        s_a_dic = {}
        for s in self.S:
            s_a_dic[tuple(s)] = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        return s_a_dic

    def initiate(self):
        v_list = {}
        s_a_dic = {}
        for s in self.S:
            v_list[tuple(s)] = 0  # v_list의 형태는 어떻게 나와야 하는가? dic 이여야 할 것 같은데? 결과값은 숫자로
            s_a_dic[tuple(s)] = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

        return v_list, s_a_dic

    def prob_foisson(self,x, lamda):
        return lamda ** x * math.e**(-lamda) / math.factorial(x)


    def alt_r(self, s, a_rent, b_rent, a): #s 상태를 받아 비교하기.
        actual_rent = [min(s[0], a_rent), min(s[1], b_rent)]
        actual_a = min((s[0] - actual_rent[0]) - (s[1] - actual_rent[1]), a)
        return sum(actual_rent)*10 - 2*actual_a

    def pi(self, s, a): # 단일 확률 값으로 계산되어야 함. dic에서 값 불러오기
        return self.S_A[tuple(s)][a+5]

    def action_pi(self, s): # 가능성이 조금이라도 있으면 행동 list로 반환
        action_lst = [index-5 for index, prob in enumerate(self.S_A[tuple(s)]) if prob > 0]
        return action_lst

    def next_s(self, s, A, a_return, b_return): # 상태 s에서 a 행동을 했을 때 이동하는 곳. 여러 가능성을 list 형태로 반환
        # A가 list로 주어지면 단일값으로 쪼개기. 그리고 결과값을 이중 리스트로 표현
        if type(A) == list :
            ans = []
            for a in A :
                ans.append(self.next_s(s,a))
            return ans

        # 단일 행동이 부여되었을 때 다음 결과값 부여
        # 입력 변수를 줄이기 위해서, s의 값은 이미 각각 대여를 끝낸 이후여야 한다.
        # 차량의 개수가 20개를 넘으면 본사로 돌려주기로 함.
        if A > 0 :
            actual_a = min(s[0], A) # 0 보다 크다
            return [min(s[0] - actual_a + a_return, 20), min(s[1] + actual_a + b_return, 20)]
        else :
            actual_a = min(s[1], -A) # 0보다 크다
            return [min(s[0] + actual_a + a_return, 20), min(s[1] - actual_a + b_return, 20)]

    def alt_p(self, a_rent,a_return,b_rent,b_return): #
        return self.prob_foisson(a_rent, self.A_rent_avg)*self.prob_foisson(a_return, self.A_return_avg)*self.prob_foisson(b_rent, self.B_rent_avg)*self.prob_foisson(b_return, self.B_return_avg)


    def r(self, next_s, a, s):
        pass

#Q. alt_r을 다시 r의 형태로 바꿀 수는 없을까? 가능할듯?
#> 일단 위의 형태로 표현한 이유는 a_rent, b_rent의 값을 알아야 제대로 r의 값을 구할 수 있기 때문에
#하지만 next_s의 값은 a_rent, a_return, b_rent, b_return, s, a의 값을 다 알아야 구할 수 있는 것
#역으로 next_s의 값을 안다고 해서 a_rent, b_rent의 값을 구할 수가 없다.


    def V(self, s): #상태 s를 입력했을 때의 가치 함수의 단일 값
        sum_all = 0
        for a in self.action_pi(s): #actino_pi 의 행동 list
            for a_rent in range(21) :
                for b_rent in range(21) :
                    r = self.alt_r(s, a_rent, b_rent, a)
                    s = [max(s[0]-a_rent, 0), max(s[1]-b_rent, 0)] # 빌려주고 남은 차량 수
                    for a_return in range(21) :
                        for b_return in range(21) :
                            next_s = self.next_s(s, a, a_return, b_return)
                            sum_all += self.pi(s,a)*self.alt_p(a_rent, a_return, b_rent, b_return)*(r+self.gamma*self.v_list[tuple(next_s)])
                        # 마지막에 next_s에 대한 상태가치함수의 값은 기존 V의 값을 가져와야 한다.
        return sum_all

    def state_evaluate(self):
        next_v_dic = self.v_list
        delta = 0
        n = 0
        while delta < self.epsilon:
            n += 1
            delta = 0
            v_dic = next_v_dic
            for s in self.S: # 수정 필요.
                next_v_dic[tuple(s)] = self.V(s)
            diff = math.sqrt(sum([(v_dic[tuple(s)] - next_v_dic[tuple(s)])**2 for s in self.S]))
            delta = max(delta, diff)
        self.count = n

        self.v_list = next_v_dic
            # v의 값이 수렴해 있겠네. 그런데 지금의 형태는 s가 바뀔 때마다 V(s)의 값이 바뀌는 구조 아닌가?
            # 각 s별로 V(s)값이 따로 저장되어 있어야 겠는데? list의 형태로. (조치완료)
            # (조치완료) 계산 한 값은 self.v_list에 저장하여 다른 곳에서도 사용할 수 있도록 설정


    def q(self, s, a):
        sum_all = 0
        for a_return in range(21) :
            for b_return in range(21) :
                for a_rent in range(21) :
                    for b_rent in range(21) :
                        change_s = [max(s[0]-a_return,0), max(s[1]-b_return, 0)]
                        next_s = self.next_s(change_s, a, a_return, b_return)
                        r = self.alt_r(s, a_rent, b_rent, a)
                        sum_all += self.pi(s, a) * self.alt_p(a_rent, a_return, b_rent, b_return) * (r + self.gamma * self.v_list[tuple(next_s)])

        return sum_all

    def improve_policy(self):
        action = self.S_A
        for s in self.S:
            before_action = action
            lst, a_lst = [], [] # 수정 필요

            for a in self.action_pi(s) :
                lst.append(self.q(s, a))
                a_lst.append(a)
            action[tuple(s)] = a_lst[np.argmax(lst)]
            # 일부 수도 코드 수정. 기존 상태라면 매번 이전행동 != action[s]를 비교해야함.

        if before_action == action  or self.count == 1:
            return self.v_list, self.S_A

        self.state_evaluate()


S = sum([[[i,j] for i in range(21)] for j in range(21)], [])
A = list(range(-5,6))

test = Policy_Evaluation(S,A)
test.improve_policy()
print(test.S_A)
print(test.v_list)
print("finish!")