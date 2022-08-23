import math
import sys

n,k = list(map(int, sys.stdin.readline().split()))
result = math.factorial(n) //(math.factorial(n-k) * math.factorial(k))
result = result % 10007

print(result)

#???? // 랑 /가 차이가 있는건가? 

from math import factorial
n, k = map(int, input().split())
result = factorial(n) // (factorial(k) * factorial(n - k))
print(result % 10007)