"""
문제
알파벳 소문자로만 이루어진 단어 S가 주어진다. 각각의 알파벳에 대해서, 단어에 포함되어 있는 경우에는 처음 등장하는 위치를, 포함되어 있지 않은 경우에는 -1을 출력하는 프로그램을 작성하시오.

입력
첫째 줄에 단어 S가 주어진다. 단어의 길이는 100을 넘지 않으며, 알파벳 소문자로만 이루어져 있다.

출력
각각의 알파벳에 대해서, a가 처음 등장하는 위치, b가 처음 등장하는 위치, ... z가 처음 등장하는 위치를 공백으로 구분해서 출력한다.
만약, 어떤 알파벳이 단어에 포함되어 있지 않다면 -1을 출력한다. 단어의 첫 번째 글자는 0번째 위치이고, 두 번째 글자는 1번째 위치이다.

"""
import string
alphabet_list = list(string.ascii_lowercase)
lst = [0]*26

s = list(input())
for alphabet in s :
    lst[alphabet_list.index(alphabet)] = s.index(alphabet)

# 0인 값들을 -1로 변경
for index, _ in enumerate(lst):
    if _ == 0 : lst[index] = -1

# index가 0이 였던 경우만 따로 값을 0으로 변환
lst[alphabet_list.index(s[0])] = 0

ans = str(lst[0])
for _ in lst[1:] :
    ans = ans +" " + str(_)
print(ans)
