"""
문제
M이상 N이하의 소수를 모두 출력하는 프로그램을 작성하시오.

입력
첫째 줄에 자연수 M과 N이 빈 칸을 사이에 두고 주어진다. (1 ≤ M ≤ N ≤ 1,000,000) M이상 N이하의 소수가 하나 이상 있는 입력만 주어진다.

출력
한 줄에 하나씩, 증가하는 순서대로 소수를 출력한다.
"""
"""
# 시간 초과 발생
import sys
A, B = sys.stdin.readline().split()
A,B = int(A), int(B)

# list 내에 있는 값들을 제거하는 형태로

lst = [i for i in range(A,B+1) if (i == 2 or i%2 != 0) and (i == 3 or i%3 != 0) and (i == 5 or i%5 != 0) and (i == 7 or i%7 != 0)]
for num in lst :
    primal = True
    if num == 1 or num == 2 or num == 3 or num == 5 or num == 7 :
        print(num)
        continue
    for n in range(11, max(11, (num//2)+1)) : # 3이상의 경우 자신의 절반보다 작은 수로 다 나눠봄.
        if num % n == 0 :
            primal = False
            break
    if primal == True : print(num)
"""

import sys
A, B = sys.stdin.readline().split()
A,B = int(A), int(B)

for num in range(A,B+1) :
    primal = True
    # 계산 속도를 줄이기 위해 소수 4개에 대해서 사전에 걸러줌. 
    if num == 1 : continue

    for n in range(2, int(num**0.5)+1) : # 3이상의 경우 자신의 절반보다 작은 수로 다 나눠봄.
        if num % n == 0 :
            primal = False
            break
    if primal == True : print(num)

