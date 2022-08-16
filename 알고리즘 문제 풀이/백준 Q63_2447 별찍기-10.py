"""
문제
재귀적인 패턴으로 별을 찍어 보자. N이 3의 거듭제곱(3, 9, 27, ...)이라고 할 때, 크기 N의 패턴은 N×N 정사각형 모양이다.

크기 3의 패턴은 가운데에 공백이 있고, 가운데를 제외한 모든 칸에 별이 하나씩 있는 패턴이다.

***
* *
***
N이 3보다 클 경우, 크기 N의 패턴은 공백으로 채워진 가운데의 (N/3)×(N/3) 정사각형을 크기 N/3의 패턴으로 둘러싼 형태이다. 예를 들어 크기 27의 패턴은 예제 출력 1과 같다.

입력
첫째 줄에 N이 주어진다. N은 3의 거듭제곱이다. 즉 어떤 정수 k에 대해 N=3k이며, 이때 1 ≤ k < 8이다.

출력
첫째 줄부터 N번째 줄까지 별을 출력한다.

"""

# 런타임 에러. 리스트냐, 이중 리스트냐는 중요하지 않았나 보네
import math
import sys

num = math.log(int(sys.stdin.readline()), 3) +1

def star(n) :
    if n == 1 : return ["*"]
    a = star(n-1) # list
    b = a + [" "*len(a)]*len(a) + a
    a = a*3
    c = [a[i] + b[i] + a[i] for i in range(len(a))]
    return c

for _ in star(num) :
    print(_)


"""
#이것은 정답 풀이. 좌우로 늘린 각 층별로 출력. list comprehension 없이 구현 가능. 
# 상하 ↔ 좌우 방향을 잘 바꿀 수 있을 것  
def draw_stars(n):
  if n==1:
    return ['*']

  Stars=draw_stars(n//3)
  L=[]

  for star in Stars:
    L.append(star*3)
  for star in Stars:
    L.append(star+' '*(n//3)+star)
  for star in Stars:
    L.append(star*3)

  return L

N=int(input())
print('\n'.join(draw_stars(N)))
"""

"""
def draw_star(n) :
    if n ==1 : return [["*"]]
    a = draw_star(n-1)
    b = a + [[" "]*len(a)]*len(a) + a
    a = a*3 # 위 아래로 복사
    c = [a[i] + b[i] + a[i] for i in range(len(a))] # 좌우로 붙이기

    return c
"""