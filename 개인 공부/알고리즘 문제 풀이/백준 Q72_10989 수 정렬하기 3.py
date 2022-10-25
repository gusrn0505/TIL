"""
문제
N개의 수가 주어졌을 때, 이를 오름차순으로 정렬하는 프로그램을 작성하시오.

입력
첫째 줄에 수의 개수 N(1 ≤ N ≤ 10,000,000)이 주어진다. 둘째 줄부터 N개의 줄에는 수가 주어진다. 이 수는 10,000보다 작거나 같은 자연수이다.

출력
첫째 줄부터 N개의 줄에 오름차순으로 정렬한 결과를 한 줄에 하나씩 출력한다.
"""

# 이대로 내면 메모리가 초과한다.
# sort 방식의 시간 복잡도는 O(N logN). 더 줄여야 함.
# defaultdict 로 이미 숫자의 순서를 지정해놓은 다음, 횟수를 구한다면 메모리 계산이 확 줄어들 수 있겠는걸
import sys
from collections import defaultdict

case_num = int(input())
dic = defaultdict(int)

for _ in range(1,10001) : dic[_] =0

for _ in range(case_num) :
    num = int(sys.stdin.readline())
    dic[num] += 1
value = [i[0] for i in list(dic.items()) if i[1] !=0]
count = [i for i in list(dic.values()) if i !=0]

for i in range(len(value)) :
    for c in range(count[i]) :
        print(value[i])
