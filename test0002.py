import numpy as np
'''
aaa = np.array([1,2,3,4,5])
print(aaa.shape)    # (5, ) 벡터

aaa = aaa.reshape(1,5)
print(aaa.shape)    # (1, 5) 행렬

bbb = np.array([[1,2,3,],[4,5,6]])
print(bbb.shape)    #(2, 3)

ccc = np.array([[1,2], [3,4], [5,6]])
print(ccc.shape)

ddd = ccc.reshape(1,3,2,1)
print(ddd.shape)
print(ddd)
'''
arr = np.arange(0, 4*2*4)
print(len(arr))

print(arr)
v = arr.reshape([4,2,4]) ## 차원 변환 [4, 2, 4]; row: 4, column: 2, depth: 4
print(v)
print(v.ndim)      ## v의 차원

print(v.sum())     ## 모든 element의 합

res01=v.sum(axis=0) ## axis=0 기준 합계
print(res01.shape)

print(res01)
res02=v.sum(axis=1)  ## axis=1 기준 합계
print(res02.shape)

print(res02)

res03=v.sum(axis=2)  ## axis=2 기준 합계
print(res03.shape)

print(res03)
