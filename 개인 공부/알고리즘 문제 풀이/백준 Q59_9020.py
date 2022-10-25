"""
문제
1보다 큰 자연수 중에서  1과 자기 자신을 제외한 약수가 없는 자연수를 소수라고 한다. 예를 들어, 5는 1과 5를 제외한 약수가 없기 때문에 소수이다. 하지만, 6은 6 = 2 × 3 이기 때문에 소수가 아니다.
골드바흐의 추측은 유명한 정수론의 미해결 문제로, 2보다 큰 모든 짝수는 두 소수의 합으로 나타낼 수 있다는 것이다. 이러한 수를 골드바흐 수라고 한다. 또, 짝수를 두 소수의 합으로 나타내는 표현을 그 수의 골드바흐 파티션이라고 한다. 예를 들면, 4 = 2 + 2, 6 = 3 + 3, 8 = 3 + 5, 10 = 5 + 5, 12 = 5 + 7, 14 = 3 + 11, 14 = 7 + 7이다. 10000보다 작거나 같은 모든 짝수 n에 대한 골드바흐 파티션은 존재한다.
2보다 큰 짝수 n이 주어졌을 때, n의 골드바흐 파티션을 출력하는 프로그램을 작성하시오. 만약 가능한 n의 골드바흐 파티션이 여러 가지인 경우에는 두 소수의 차이가 가장 작은 것을 출력한다.

입력
첫째 줄에 테스트 케이스의 개수 T가 주어진다. 각 테스트 케이스는 한 줄로 이루어져 있고 짝수 n이 주어진다.

출력
각 테스트 케이스에 대해서 주어진 n의 골드바흐 파티션을 출력한다. 출력하는 소수는 작은 것부터 먼저 출력하며, 공백으로 구분한다.
"""

# 시간 초과 발생
import math
case_num = int(input())

# 소수들의 집합을 통해서 골드바그 합 구하기

case_lst = []
for _ in range(case_num) :
    num = int(input())
    case_lst.append(num)

m = max(case_lst)
array1 = [True for _ in range(m + 1)]  # 소수 판별을 위한 리스트설정(True면 소수)
array1[0], array1[1] = False, False  # 0,1은 소수가 아니기에 False로 설정
for i in range(2, int(math.sqrt(m) + 1)):
    if array1[i]:
        j = 2
        while i * j <= m:
            array1[i * j] = False
            j += 1

# 이전 문제의 논리를 차용하여 주어진 case 중 가장 큰 수를 기반으로 소수 list 생성
lst = [index for index, value in enumerate(array1) if value == True]


from collections import defaultdict
ans_lst = defaultdict(list)
for num in case_lst :

    # 주어진 숫자 num//2 보다 작거나 같은 소수들의 개수 확인하기.
    #해당 소수들에 대해서만 for 절로 골드바드 합이 되는 쌍들 알아보기

    #아래 조건 식을 좀 더 간단히 할 수 있겠는걸. 일단 보류
#    count_primal = len([primal for primal in lst if primal <= num//2])
    dic = {}
    for _ in lst : dic[num-_] = _
    for first in lst :
        try : ans_lst[num].append([first, dic[first]])
        except : continue

    # 경우의 수가 1개이면 바로 출력하기
    if len(ans_lst[num]) == 1 : print(ans_lst[num][0][0], ans_lst[num][0][1])
    # 경우의 수가 여러개면, 두 소수의 차를 구한 diff_lst를 통해서 최소 쌍 알아내기
    else :
        diff_lst = [abs(s[1]-s[0]) for s in ans_lst[num]] #여기서 s는 list임.
        ans = ans_lst[num][diff_lst.index(min(diff_lst))]
        print(ans[0], ans[1])


