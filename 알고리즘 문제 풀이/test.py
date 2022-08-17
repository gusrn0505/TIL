# TD(0) 예측 코드 참고
from collections import defaultdict
import random
import numpy as np

# 외부용 함수 reward_func 간략 구현 (이전 예시 활용)
def reward_func(next_s, a, s):
    # next_s 와 s의 차이가 짝수이면 +1, 홀수면 -1
    # 단, a의 크기에 반비례함.
    if abs(next_s - s) % 2 == 0:
        reward = 1
    else:
        reward = -1

    if a == 0:
        return 0
    else:
        return reward / a  # 즉, a가 양수이며 짝수이며, 가능한 작을 때 (=2) 일 때 최대의 보상이 주어지도록 설정


# 최대값이 2개 이상인 경우, 임의로 1개의 최대값을 만들어낸 행동 a를 산출
def choose_random_max(lst):
    max_arg = np.where(np.array(lst) >= max(lst))
    return random.choice(list(max_arg)[0])  # max_arg가 array 형태로 안에 있는 list를 꺼내기 위해 [0] 사용


class evaluate_Q_TD:
    def __init__(self, S, A, reward_func, alpha=0.1, epsilon=0.001, gamma=0.9, num_episode=10, len_episode=20):
        self.S = S
        self.A = A
        self.reward_func = reward_func
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_episode = num_episode
        self.T = len_episode
        self.Q = self.initiate_Q()

    #        self.b, self.C = self.initiate_b()
    #        self.pi = pi
    #        self.V = self.initiate_V()

    def initiate_Q(self):  # Q(s,a) 값을 초기화
        Q_dict = defaultdict(float)
        for s in self.S:
            for a in self.A:
                Q_dict[(s, a)] = 0

        return Q_dict

    def pi(self, s):  # Q 값을 기반해서 가장 가치가 높은 행동 a 산출
        lst = []
        for a in self.A: lst.append(self.Q[(s, a)])
        return choose_random_max(lst)

    def choice_action(self, s, policy):  # 일반화. 정책 기반으로 상황 s에 있을 때 선택할 행동 a 산출
        policy_a_list = []
        for _ in self.A:
            policy_a_list.append(policy(s, _))

        a = random.choices(self.A, weights=policy_a_list)
        a = a[0]
        return a

    def next_s(self, s, a):  # 상태 s에서 a 행동을 했을 때 다음 상태 s'. 정책, S,A 에 따라 달라짐.
        return min(max(s + a, 0), max(self.S))

    def make_episode(self, start_s, T):
        s = start_s
        episode = {"S": [], "A": [], "R": []}
        episode["R"].append(0)  # R_0 값 부여
        for _ in range(T):
            episode["S"].append(s)
            a = self.pi(s)  # pi 함수가 결정론적으로 a 값을 반환함에 따라 수정
            next_s = self.next_s(s, a)
            r = self.reward_func(next_s, a, s)
            episode["A"].append(a)
            episode["R"].append(r)
            s = next_s
        return episode

    def update_returns(self):  # Q 추정 및 제어를 위해 수정

        for _ in range(self.num_episode):
            start_s = random.choice(self.S)  # 시작 탐험 가정
            episode = self.make_episode(start_s, self.T)
            S, R, A = episode['S'], episode['R'], episode['A']
            # make_episode 에서 이미 a,r를 계산해 두었기 때문에 Q(s,a)만 갱신하겠음.
            for index, s in enumerate(S[:-1]):
                index_origin_s = self.S.index(s)
                next_s = S[index + 1]
                self.Q[(s, R[index])] = self.Q[(s, R[index])] + self.alpha * (
                    R[index + 1] + self.gamma * self.Q[(next_s, A[index + 1])] - self.Q[(s, R[index])])
