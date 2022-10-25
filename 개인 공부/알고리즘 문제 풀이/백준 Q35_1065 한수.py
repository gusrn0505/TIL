"""
문제
어떤 양의 정수 X의 각 자리가 등차수열을 이룬다면, 그 수를 한수라고 한다. 등차수열은 연속된 두 개의 수의 차이가 일정한 수열을 말한다. N이 주어졌을 때, 1보다 크거나 같고, N보다 작거나 같은 한수의 개수를 출력하는 프로그램을 작성하시오.

입력
첫째 줄에 1,000보다 작거나 같은 자연수 N이 주어진다.

출력
첫째 줄에 1보다 크거나 같고, N보다 작거나 같은 한수의 개수를 출력한다.
"""
# 각 모든 경우를 한수 인지 체크하면 시간 초과가 일어날 것으로 봄.
# 따라서 정답 list를 조건에 맞춰 형성한 다음, filter로 개수를 셈.

a_lst = list(range(1,100))
odd = list(range(1,10,2))
even = list(range(2,10,2))

def make_lst(lst) :
    result = []
    for a in lst :
        for b in lst :
            new_num = str(a)+ str(int((a+b)/2)) +str(b)
            result.append(int(new_num))
    return result

odd_list = make_lst(odd)
even_list = make_lst(even)
even_list.extend([210, 420, 630, 840])

ans_list = a_lst +odd_list + even_list
ans_list.sort()

num = int(input())
ans_lst = [i for i in ans_list if i <= num]
print(len(ans_lst))