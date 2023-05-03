import faiss
import numpy as np

dummy_array = np.random.random((5, 8))
query_vector = np.random.random((1, 8))

hnsw_index = faiss.IndexHNSWFlat(8, 3)
# hnsw_index.train(dummy_array)
hnsw_index.add(dummy_array)
x = hnsw_index.search(query_vector, k=2)
i = 0