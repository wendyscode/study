def multiply(x):
    return x * 10
 
numbers = [1,2,3,4]

result = map(multiply,numbers)

print(result)
#<map object at 0x0000019351E0A9E0>

print(list(result))
