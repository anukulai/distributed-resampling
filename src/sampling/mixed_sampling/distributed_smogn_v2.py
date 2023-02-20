import logging
import math

import numpy as np
import pandas as pd
import random
from pyspark import keyword_only
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import faiss
from sklearn.metrics import euclidean_distances

from src.params.sampling._smogn import _KMeansParams, _SMOGNParams
from src.sampling.mixed_sampling.base import BaseMixedSampler
from src.utils.dataframe import get_num_cols, get_cat_cols

logging.basicConfig(level=20)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=20)


def is_single_partition_present(df):
    """
    This function checks for presence of single valued partitions.
    :param df: dataframe with partitions
    :return: True if the there is a presence of a partition with 1 count.
    """
    candidacy_dict = df.partition.value_counts().to_dict()

    for partition_key, candidacy in candidacy_dict.items():
        if candidacy == 1:
            return True

    return False

class DistributedSMOGN_v2(BaseMixedSampler, _KMeansParams, _SMOGNParams):
    @keyword_only
    def __init__(self, label_col=None, sampling_strategy="balance", k_partitions=2, threshold=0.8, method="auto",
                 xtrm_type="both", coef=1.5, ctrl_pts_region=None, init_steps=2, tol=1e-4, max_iter=20, k_neighbours=5,
                 perturbation=0.02):
        super(DistributedSMOGN_v2, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, label_col=None, sampling_strategy="balance", k_partitions=2, threshold=0.8, method="auto",
                  xtrm_type="both", coef=1.5, ctrl_pts_region=None, init_steps=2, tol=1e-4, max_iter=20, k_neighbours=5,
                  perturbation=0.02):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _partition(self, df, partition_col):
        feature_vector_col = "feature_vector"
        feature_vector_cols = get_num_cols(df)

        redo_partition = True

        while redo_partition:
            df_new = df
            df_new = VectorAssembler(inputCols=feature_vector_cols, outputCol=feature_vector_col).transform(df_new)
            df_new = KMeans(
                featuresCol=feature_vector_col, predictionCol=partition_col, k=self.getKPartitions(),
                initSteps=self.getInitSteps(), tol=self.getTol(), maxIter=self.getMaxIter(),
                seed=random.randint(0, 1000)
            ).fit(df_new).transform(df_new)

            df_new = df_new.drop(feature_vector_col)
            redo_partition = is_single_partition_present(df_new.toPandas())

        return df_new.repartition(self.getKPartitions(), partition_col)

    def _oversample(self, bump):
        schema = bump.samples.schema

        # Broadcast shared variables to all partitions
        sc = bump.samples.rdd.context

        cat_feature_cols = sc.broadcast(get_cat_cols(bump.samples.drop(self.getLabelCol())))
        num_feature_cols = sc.broadcast(get_num_cols(bump.samples.drop(self.getLabelCol())))
        label_col = sc.broadcast(self.getLabelCol())
        n_synth_samples = sc.broadcast(round(bump.sampling_percentage))
        k_neighbours = sc.broadcast(self.getKNeighbours())
        perturbation = sc.broadcast(self.getPerturbation())

        partition_col = "partition"
        bump.samples = self._partition(bump.samples, partition_col)

        def create_synth_samples(partition):
            synth_samples = self._create_synth_samples(
                partition=partition,
                cat_feature_cols=cat_feature_cols.value,
                num_feature_cols=num_feature_cols.value,
                label_col=label_col.value,
                n_synth_samples=n_synth_samples.value,
                k=k_neighbours.value,
                perturbation=perturbation.value
            )

            return pd.DataFrame(data=synth_samples)

        # return bump.samples.groupby(partition_col).applyInPandas(create_synth_samples, schema=schema)
        return bump.samples.toPandas().groupby(partition_col).apply(create_synth_samples)

    def _undersample(self, bump):
        return super()._partition(bump.samples).sample(withReplacement=False, fraction=bump.sampling_percentage)

    def _create_synth_sample_SMOTE(self, base_sample, neighbour_sample, cat_feature_cols, num_feature_cols, label_col,
                                   base_sample_feature_vector, neighbour_sample_feature_vector):
        synth_sample_cat_features = {
            cat_feature_col: np.random.choice([base_sample[cat_feature_col], neighbour_sample[cat_feature_col]])
            for cat_feature_col in cat_feature_cols
        }

        synth_sample_num_features = {
            num_feature_col: base_sample[num_feature_col] + abs(
                (neighbour_sample[num_feature_col] - base_sample[num_feature_col])) * np.random.uniform(0, 1)
            for num_feature_col in num_feature_cols
        }

        synth_sample_feature_vector = np.asarray(list(synth_sample_num_features.values()))

        base_sample_dist = np.linalg.norm(synth_sample_feature_vector - base_sample_feature_vector)
        neighbour_sample_dist = np.linalg.norm(synth_sample_feature_vector - neighbour_sample_feature_vector)

        synth_sample_label = {
            label_col: (base_sample[label_col] + neighbour_sample[label_col]) / 2
            if base_sample_dist == neighbour_sample_dist
            else (neighbour_sample_dist * base_sample[label_col] + base_sample_dist * neighbour_sample[label_col]) / (
                    base_sample_dist + neighbour_sample_dist)
        }

        return {**synth_sample_cat_features, **synth_sample_num_features, **synth_sample_label}

    def _create_synth_sample_GN(self, base_sample, cat_feature_cols, num_feature_cols, label_col, cat_feature_probs,
                                num_feature_stds, label_std, perturbation):
        synth_sample_cat_features = {
            cat_feature_col: np.random.choice(list(cat_feature_probs[cat_feature_col].keys()),
                                              p=list(cat_feature_probs[cat_feature_col].values()))
            for cat_feature_col in cat_feature_cols
        }

        synth_sample_num_features = {
            num_feature_col: base_sample[num_feature_col] + np.random.normal(0, num_feature_stds[
                num_feature_col] * perturbation)
            for num_feature_col in num_feature_cols
        }

        synth_sample_label = {
            label_col: base_sample[label_col] + np.random.normal(0, label_std * perturbation)
        }

        return {**synth_sample_cat_features, **synth_sample_num_features, **synth_sample_label}

    def _create_synth_samples(self, partition, cat_feature_cols, num_feature_cols, label_col, n_synth_samples, k,
                              perturbation):
        n_rows = len(partition.index)
        k = min(k, n_rows)

        # Calculate feature vectors
        feature_vectors = partition[[*num_feature_cols]].to_numpy()

        # Using IVFPQ to get k nearest neighbours
        n_dim = feature_vectors.shape[1]
        # n_dim = feature_vectors.size()  # Dimension (length) of vectors.

        #Calculating the number of clusters based on parrtition size
        nlist = self.get_num_clusters(n_rows)  # Number of inverted lists (number of partitions or cells).

        #calculate nsegments
        #nsegment = self.

        nsegment = self.get_num_segments(n_dim) # Number of segments for product quantization (number of subquantizers).
        nbit = 8  # Number of bits to encode each segment.

        # xt = np.random.random((500000, 20))

        # Create the index.
        coarse_quantizer = faiss.IndexFlat(n_dim)
        index = faiss.IndexIVFPQ(coarse_quantizer, n_dim, nlist, nsegment, nbit)

        index.train(feature_vectors)
        index.add(feature_vectors)
        index.nprobe = 10
        # Calculate euclidean distances between various combination of rows.
        # dist_matrix = euclidean_distances(feature_vectors, feature_vectors)

        # This is an n_rows by k matrix. Each row represents a point with indices to its k nearest neighbours.
        # neighbour_sample_index_matrix = np.delete(np.argsort(dist_matrix, axis=1), np.s_[k + 1:], axis=1)

        # a dictionary representing the distribution of categorical features
        cat_feature_probs = {
            cat_feature_col: partition[cat_feature_col].value_counts(normalize=True).to_dict()
            for cat_feature_col in cat_feature_cols
        }

        # standard deviations of all the numerical features in partition
        num_feature_stds = partition[[*num_feature_cols]].std()

        # standard deviation of the target column
        label_std = partition[label_col].std()

        # placeholder for the number of synthetic samples ---> n_rows are incremented by a factor of n_synth_samples
        synth_samples = [None for _ in range(n_rows * n_synth_samples)]

        for base_sample_index, base_sample in partition.iterrows():
            if base_sample_index % 500 == 0:
                LOGGER.info(f"Processing the {base_sample_index}th data point")
            # Iterate through the partition and fetch distances and neighbour indices for a particular row.
            query_vector = base_sample[[*num_feature_cols]].to_numpy().reshape((1, n_dim))

            # LOGGER.info("Searching the Index Now!")
            dists, neighbour_sample_indices = index.search(query_vector, k)
            dists = dists[0]  # unpack it to reshape
            neighbour_sample_indices = neighbour_sample_indices[0]  # unpack it to reshaope
            if base_sample_index % 500 == 0:
                LOGGER.info(
                    f"Found the {k} nearest neighbours with distances: {dists} and indices: {neighbour_sample_indices}"
                )
            # dists = dist_matrix[base_sample_index]

            for n_synth_sample in range(n_synth_samples):

                # pick any random neighbour index
                random_num = random.randint(0, k - 1)
                neighbour_sample_index = neighbour_sample_indices[random_num]
                neighbour_sample = partition.iloc[neighbour_sample_index]

                dist = dists[random_num]
                # safe_dist = dists[neighbour_sample_indices[(k + 1) // 2]] / 2
                safe_dist = dists[(k + 1) // 2] / 2

                if dist < safe_dist:
                    synth_sample = self._create_synth_sample_SMOTE(
                        base_sample=base_sample,
                        neighbour_sample=neighbour_sample,
                        cat_feature_cols=cat_feature_cols,
                        num_feature_cols=num_feature_cols,
                        label_col=label_col,
                        base_sample_feature_vector=feature_vectors[base_sample_index],
                        neighbour_sample_feature_vector=feature_vectors[neighbour_sample_index]
                    )

                else:
                    synth_sample = self._create_synth_sample_GN(
                        base_sample=base_sample,
                        cat_feature_cols=cat_feature_cols,
                        num_feature_cols=num_feature_cols,
                        label_col=label_col,
                        cat_feature_probs=cat_feature_probs,
                        num_feature_stds=num_feature_stds,
                        label_std=label_std,
                        perturbation=min(safe_dist, perturbation)
                    )

                synth_samples[base_sample_index * n_synth_samples + n_synth_sample] = synth_sample

        return synth_samples

    def get_num_clusters(self, partition_size):
        if partition_size <= 100:      #size: 0-100
            return 1

        elif 100 < partition_size <= 1000:    #size: 20-200
            return 5

        elif 1000 < partition_size <= 10000:      #size:100-1000
            return 10

        elif 10000 < partition_size <= 50000:      #size:500-2500
            return 20

        elif 50000 < partition_size <= 100000:    #size:1000-2000
            return 50

        else:
            return 50

    def get_num_segments(self,vector_size):
        if vector_size <= 30:
            if vector_size % 5 == 0:
                return 5
            elif vector_size % 4 == 0:
                return 4
            elif vector_size % 3 == 0:
                return 3
            elif vector_size % 2 == 0:
                return 2
            else:
                return  vector_size

        else:
            if vector_size % 10 == 0:
                return 10
            elif vector_size % 9 == 0:
                return 9
            elif vector_size % 8 == 0:
                return 8
            elif vector_size % 7 == 0:
                return 7
            elif vector_size % 6 == 0:
                return 6
            else:
                return vector_size




