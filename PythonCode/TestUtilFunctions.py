def has_duplicates(array) -> bool:
    from collections import Counter
    # Count the number of occurrences of each element in the array
    count = Counter(array)
    # Check if any element has a count greater than 1
    return any(count[x] > 1 for x in count)