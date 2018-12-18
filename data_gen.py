vector_size = 6
data_str = 'x = np.matrix([\n'
for i in range(2**vector_size):
    data_str = data_str + '    ['
    for j in range(vector_size-1,-1,-1):
        if j == 0:
            end = ''
        else:
            end = ','

        if (i >> j) % 2:
            data_str = data_str + '1' + end
        else:
            data_str = data_str + '0' + end
    data_str = data_str + '],\n'
data_str = data_str + '])\n'

data_str = data_str + 'y = np.matrix([\n'
for i in range(2**vector_size):
    ans = 0
    for j in range(vector_size//2):
        if (i >> j) % 2 != ( i >> (vector_size - 1 - j)) % 2:
            ans = 1
            break
    data_str = data_str + '    [{}],\n'.format(ans)
data_str = data_str + '])'

print(data_str)