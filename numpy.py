import numpy as np

a = np.array([1,2,3])
b = np.random.normal(loc=0.0, scale=1.0)
np.mean(a)
np.std(a)
np.prod([1,2,3,4])
np.reshape((3, 2))
a.reshape((-1, 3))

# Save an array to a binary file in NumPy .npy format
np.save('/tmp/123', np.array([[1, 2, 3], [4, 5, 6]]))
np.load('/tmp/123.npy')
np.loadtxt

# Save several arrays into a single file in uncompressed .npz format.
np.savez(file, *args, **kwds)

# for 2-D arrays:
#     - np.hstack
#     - np.vstack
#     - np.column_stack
#     - np.row_stack
# for 3-D arrays (the above plus):
#     - np.dstack
# for N-D arrays:
#     - np.concatenateargument list
