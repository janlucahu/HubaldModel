from calculations import *

#parameters, constants = initialize(1000, (200_000, 2_000_000), 0.5)
#sparse = sparse_prob_matrix(parameters, constants, 2000, 3)

#print(sparse)

# Create a sample 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]], dtype=float)

mask = np.any(arr == 5, axis=1) | np.any(arr == 8, axis=1)
arr = arr[~mask]
newRow = np.array([10, 11, 20])
arr = np.vstack((arr, newRow))
print(mask)
ii = 0
for val, jj in enumerate(mask):
    print(val, jj)
print(ii)
