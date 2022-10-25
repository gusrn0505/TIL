"""
문제
셀프 넘버는 1949년 인도 수학자 D.R. Kaprekar가 이름 붙였다. 양의 정수 n에 대해서 d(n)을 n과 n의 각 자리수를 더하는 함수라고 정의하자. 예를 들어, d(75) = 75+7+5 = 87이다.
양의 정수 n이 주어졌을 때, 이 수를 시작해서 n, d(n), d(d(n)), d(d(d(n))), ...과 같은 무한 수열을 만들 수 있다.
예를 들어, 33으로 시작한다면 다음 수는 33 + 3 + 3 = 39이고, 그 다음 수는 39 + 3 + 9 = 51, 다음 수는 51 + 5 + 1 = 57이다. 이런식으로 다음과 같은 수열을 만들 수 있다.

33, 39, 51, 57, 69, 84, 96, 111, 114, 120, 123, 129, 141, ...
n을 d(n)의 생성자라고 한다. 위의 수열에서 33은 39의 생성자이고, 39는 51의 생성자, 51은 57의 생성자이다. 생성자가 한 개보다 많은 경우도 있다. 예를 들어, 101은 생성자가 2개(91과 100) 있다.
생성자가 없는 숫자를 셀프 넘버라고 한다. 100보다 작은 셀프 넘버는 총 13개가 있다. 1, 3, 5, 7, 9, 20, 31, 42, 53, 64, 75, 86, 97
10000보다 작거나 같은 셀프 넘버를 한 줄에 하나씩 출력하는 프로그램을 작성하시오.

입력
입력은 없다.

출력
10,000보다 작거나 같은 셀프 넘버를 한 줄에 하나씩 증가하는 순서로 출력한다.
"""

#시간 초과 발생
"""
시간 초과를 해결할 방법 
1. def magic_num을 좀 더 단순화한다 
2. for, while 절을 하나로 뭉친다. 
3. 해석학적으로 풀이한다. 

"""


def magic_num(a) :
    magic_num = a
    for n in str(a) : magic_num += int(n)
    return magic_num

st = set(range(1,10000))
remove_set = set()
for num in st :
    remove_set.add(magic_num(num))
self_number = st - remove_set
for self_num in sorted(self_number) :
    print(self_num)


"""
시간 초과 발생 

lst = list(range(1,10001))
for num in lst :
    start = num
    while start <= 10000 :
        start = magic_num(start)
        try : lst.remove(start) 
        # 여기서 이미 뺴진 값들을 다시 한번 뺴는 작업이 들어감.
        except : pass
        start = magic
    print(num)

"""