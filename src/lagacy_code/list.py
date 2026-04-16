list_ja = [1,2,3,4,5, 6]
filtered = [list_ja[i] for i in range(0, len(list_ja)) if list_ja[i] %2 == 0]
print(filtered)