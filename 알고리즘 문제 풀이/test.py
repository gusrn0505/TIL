import math
def draw_star(n) :
    if n ==1 : return [["*"]]
    a = draw_star(n-1)
    b = a + [[" "]*len(a)]*len(a) + a
    a = a*3
    c = [a[i] + b[i] + a[i] for i in range(len(a))]

    return c

num = math.log(int(input()), 3) +1

for _ in draw_star(num) :
    print("".join(_))

"""
b = a + [[" "]*len(a)]*len(a) + a
a = a*3
c = [a[i] + b[i] + a[i] for i in range(len(a))]

print(a)
print(b)
# 이중 리스트에 곱하기를 하면 밑으로 복사가 되고,

for _ in c :
    print("".join(_))
"""