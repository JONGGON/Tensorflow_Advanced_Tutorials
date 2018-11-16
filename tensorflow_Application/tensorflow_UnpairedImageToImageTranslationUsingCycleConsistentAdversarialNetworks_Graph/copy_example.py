# coding: utf-8
import copy

import numpy as np

# case 1
print("# a가 numpy의 array인 경우")
a = np.ones((3, 3, 3, 3))

b = a[:]
c = np.copy(a)
d = copy.copy(a)
e = list(a)
f = copy.deepcopy(a)

print("<<< 1. a전체가 copy되므로 id 값은 다 바뀔 것이다.>>>\n")
print("# a의 id : {}".format(id(a)))
print("# [:]을 이용한 shallow copy -> b의 id : {}".format(id(b)))
print("# numpy.copy를 이용한 shallow copy -> c의 id : {}".format(id(c)))
print("# copy.copy를 이용한 shallow copy -> d의 id : {}".format(id(d)))
print("# list를 이용한 shallow copy -> e의 id : {}".format(id(e)))
print("# copy.deepcopy를 이용한 deep copy -> f의 id : {}\n\n".format(id(f)))

print("<<< 2. a전체가 아닌 요소들에 접근한다면? >>>\n")
print("a[0]의 id : {}".format(id(a[0])))
print("b[0]의 id : {}".format(id(b[0])))
print("c[0]의 id : {}".format(id(c[0])))
print("d[0]의 id : {}".format(id(d[0])))
print("e[0]의 id : {} - numpy에서는 이게 deepcoppy!".format(id(e[0])))
print("f[0]의 id : {} - numpy에서는 deepcopy가 안됩니다".format(id(f[0])))
print("id(f[0]) == id(a[0])은 {}".format(str(id(f[0]) == id(a[0]))))
print("\n\n")

# case 2
print("# a가 list인 경우")
a = list([[3, 3], [3, 3]])

b = a[:]
c = np.copy(a)
d = copy.copy(a)
e = list(a)
f = copy.deepcopy(a)

print("<<< 1. a전체가 copy되므로 id 값은 다 바뀔 것이다.>>>\n")
print("# a의 id : {}".format(id(a)))
print("# [:]을 이용한 shallow copy -> b의 id : {}".format(id(b)))
print("# numpy.copy를 이용한 shallow copy -> c의 id : {}".format(id(c)))
print("# copy.copy를 이용한 shallow copy -> d의 id : {}".format(id(d)))
print("# list를 이용한 shallow copy -> e의 id : {}".format(id(e)))
print("# copy.deepcopy를 이용한 deep copy -> f의 id : {}\n\n".format(id(f)))

print("<<< 2. a전체가 아닌 요소들에 접근한다면? >>>\n")
print("a[0]의 id : {}".format(id(a[0])))
print("b[0]의 id : {}".format(id(b[0])))
print("c[0]의 id : {} - list에서는 numpy.copy가 deepcopy!!!".format(id(c[0])))
print("d[0]의 id : {}".format(id(d[0])))
print("e[0]의 id : {} - 예상대로 shollow copy".format(id(e[0])))
print("f[0]의 id : {} - 예상대로 deep copy".format(id(f[0])))
print("id(f[0]) == id(a[0])은 {}".format(str(id(f[0]) == id(a[0]))))
