"""
문제
 
$n \choose m$의 끝자리 $0$의 개수를 출력하는 프로그램을 작성하시오.

입력
첫째 줄에 정수 $n$, $m$ ($0 \le m \le n \le 2,000,000,000$, $n \ne 0$)이 들어온다.

출력
첫째 줄에
$n \choose m$의 끝자리 $0$의 개수를 출력한다.
"""

# 시간 초과 발생. Why?
import sys

n,m = list(map(int,sys.stdin.readline().split()))

def two_count(n) :
    count = 0
    while n != 0 :
        n = n //2
        count += n
    return count

def five_count(n):
    count = 0
    while n != 0 :
        n = n //5
        count += n
    return count


print(min(two_count(n)-two_count(n-m)-two_count(m), five_count(n)-five_count(n-m)- five_count(m)))