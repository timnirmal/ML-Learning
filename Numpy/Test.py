def select_second(L):
    """Return the second element of the given list. If the list has no second
    element, return None.
    """
    if len(L[0]) == 0:
        return None
    else:
        return L[0][1]


L = [[1, 2, 3]]
print(select_second(L))
