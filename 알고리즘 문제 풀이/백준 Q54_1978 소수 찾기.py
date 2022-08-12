"""
문제
주어진 수 N개 중에서 소수가 몇 개인지 찾아서 출력하는 프로그램을 작성하시오.

입력
첫 줄에 수의 개수 N이 주어진다. N은 100이하이다. 다음으로 N개의 수가 주어지는데 수는 1,000 이하의 자연수이다.

출력
주어진 수들 중 소수의 개수를 출력한다.
"""
case_num = int(input())
sum_all = 0
lst = input().split()

for num in lst :
    num = int(num)
    primal = 1
    for n in range(2, max(3, (num//2)+1)) : # 3이상의 경우 자신의 절반보다 작은 수로 다 나눠봄.
        if num % n == 0 :
            primal = 0
            break
    if num == 2 : primal = 1 # 2인 경우 추가.
    if num == 1 : primal =0
    sum_all += primal

print(sum_all)

