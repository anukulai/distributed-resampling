from abc import ABC, abstractmethod
from numbers import Number

from pyspark.ml import Transformer

from src.params._common import _LabelColParam
from src.params.sampling._base import _BaseSamplerParams
from src.utils.bump import collect_bumps
from src.utils.dataframe import init_df
from src.utils.phi import collect_phi
from src.utils.sampling import SAMPLING_METHODS


class BaseSampler(ABC, Transformer, _LabelColParam, _BaseSamplerParams):
    def _validate_relevance(self, dataset):
        if self.getThreshold() <= 0 or self.getThreshold() >= 1:
            raise ValueError(
                f"threshold should be greater than 0 and less than 1. "
                f"Got {self.getThreshold()} instead."
            )

        phi = collect_phi(
            dataset=dataset.sort(self.getLabelCol()),
            label_col=self.getLabelCol(),
            method=self.getMethod(),
            xtrm_type=self.getXtrmType(),
            coef=self.getCoef(),
            ctrl_pts_region=self.getCtrlPtsRegion()
        )

        if all(val == 0 for val in phi) or all(val == 1 for val in phi):
            raise ValueError(
                f"Invalid relevance function. "
                f"All phi values are {phi[0]}"
            )

        bumps = collect_bumps(
            dataset=dataset,
            phi=phi,
            threshold=self.getThreshold()
        )

        if len(bumps) <= 1:
            raise ValueError(
                f"Invalid relevance threshold. "
                f"{len(bumps)} defined."
            )

        return bumps

    def _validate_sampling_strategy(self, bumps):
        if isinstance(self.getSamplingStrategy(), Number):
            bumps = self._validate_sampling_strategy_number(bumps)
        elif isinstance(self.getSamplingStrategy(), list):
            bumps = self._validate_sampling_strategy_list(bumps)
        elif isinstance(self.getSamplingStrategy(), str):
            if self.getSamplingStrategy() not in SAMPLING_METHODS:
                raise ValueError(
                    f"When 'sampling_strategy' is a string, it should be one of {SAMPLING_METHODS}. "
                    f"Got {self.getSamplingStrategy()} instead."
                )

            if self.getSamplingStrategy() == "balance":
                bumps = self._validate_sampling_strategy_balance(bumps)
            else:
                bumps = self._validate_sampling_strategy_extreme(bumps)

        return bumps

    @abstractmethod
    def _validate_sampling_strategy_number(self, bumps):
        raise NotImplementedError()

    @abstractmethod
    def _validate_sampling_strategy_list(self, bumps):
        raise NotImplementedError()

    @abstractmethod
    def _validate_sampling_strategy_balance(self, bumps):
        raise NotImplementedError()

    @abstractmethod
    def _validate_sampling_strategy_extreme(self, bumps):
        raise NotImplementedError()

    def _partition(self, df, partition_col=None):
        return df.repartition(self.getKPartitions())

    @abstractmethod
    def _resample(self, bump):
        raise NotImplementedError()

    def _transform(self, dataset):
        bumps = self._validate_relevance(dataset)
        bumps = self._validate_sampling_strategy(bumps)

        new_dataset = init_df(dataset.schema)

        for bump in bumps:
            new_dataset = new_dataset.union(self._resample(bump))

        return new_dataset
