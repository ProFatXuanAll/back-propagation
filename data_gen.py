vector_size = 6
print('x = np.matrix([')
for i in range(2**vector_size):
    print('    [', end='')
    for j in range(vector_size-1,-1,-1):
        if j == 0:
            end = ''
        else:
            end = ','

        if (i >> j) % 2:
            print('1', end=end)
        else:
            print('0', end=end)
    print('],')
print('])')

print('y = np.matrix([')
for i in range(2**vector_size):
    ans = 1
    for j in range(vector_size//2):
        if (i >> j) % 2 != ( i >> (vector_size - 1 - j)) % 2:
            ans = 0
            break
    print('    [{}],'.format(ans))
print('])')