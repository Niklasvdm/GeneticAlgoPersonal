import numpy as np

def has_duplicates(array) -> bool:
    from collections import Counter
    # Count the number of occurrences of each element in the array
    count = Counter(array)
    # Check if any element has a count greater than 1
    return any(count[x] > 1 for x in count)


def fitness_sharing(fitness: np.array, sigma: float) -> np.array:
    # Calculate the Jaccard distance matrix
    distance_matrix = np.zeros((fitness.size, fitness.size))
    for i in range(fitness.size):
        for j in range(i + 1, fitness.size):
            intersection = np.intersect1d(fitness[i], fitness[j]).size
            union = np.union1d(fitness[i], fitness[j]).size
            distance_matrix[i, j] = distance_matrix[j, i] = 1 - intersection / union

    # Calculate the shared fitness values
    shared_fitness = np.zeros(fitness.size)
    for i in range(fitness.size):
        sum_share = 0
        for j in range(fitness.size):
            if i != j:
                sum_share += np.exp(-distance_matrix[i, j] ** 2 / (2 * sigma ** 2))
        shared_fitness[i] = fitness[i] / sum_share

    return shared_fitness