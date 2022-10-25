"""
문제
수직선 위에 N개의 좌표 X1, X2, ..., XN이 있다. 이 좌표에 좌표 압축을 적용하려고 한다.

Xi를 좌표 압축한 결과 X'i의 값은 Xi > Xj를 만족하는 서로 다른 좌표의 개수와 같아야 한다.
X1, X2, ..., XN에 좌표 압축을 적용한 결과 X'1, X'2, ..., X'N를 출력해보자.

입력
첫째 줄에 N이 주어진다.
둘째 줄에는 공백 한 칸으로 구분된 X1, X2, ..., XN이 주어진다.

출력
첫째 줄에 X'1, X'2, ..., X'N을 공백 한 칸으로 구분해서 출력한다.
"""

# 시간 초과 발생. 어디를 고쳐야 할까.
import sys

case_num = int(input())
num_list = list(map(int, sys.stdin.readline().split()))

lst = sorted(list(set(num_list)))
dic = {}
for index, num in enumerate(lst) :
    dic[num] = index # 역시 dict 가 짱인가
s = str(dic[num_list[0]])
for num in num_list[1:] :
    s += " " + str(dic[num])
print(s)