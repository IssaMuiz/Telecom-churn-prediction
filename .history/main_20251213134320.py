def add(a, b):
    """Adding two interger"""
    return a + b


def divide(a, b):
    """Diving two integer"""
    if b == 0:
        raise ValueError("You can't divide by zero")
    else:
        return a / b


c = divide(4, 2)
print(c)
