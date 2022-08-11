"""
백준 Q2_1000, 1001번, 10998번, 1008번 등등

문제
두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.

입력
첫째 줄에 A와 B가 주어진다. (0 < A, B < 10)
Q. 이 입력 부분을 어떻게 정의하라는 걸까?
=> A. iuput() 함수를 사용해야 한다.

출력
첫째 줄에 A+B를 출력한다.
A. 출력할 때, A와 B를 int로 변경해야 한다.
"""
A, B = input().split()
print(int(A)+int(B))
print(int(A) - int(B))
print(int(A) * int(B))
print(int(A) // int(B))
print(int(A) % int(B))

