from abc import abstractmethod

from src.sampling.base import BaseSampler


class BaseMixedSampler(BaseSampler):
    def _validate_sampling_strategy_number(self, bumps):
        raise ValueError(
            f"'sampling_strategy' cannot be a number for mixed-samplers."
        )

    def _validate_sampling_strategy_list(self, bumps):
        if len(self.getSamplingStrategy()) != len(bumps):
            raise ValueError(
                f"When 'sampling_strategy' is a list, the number of sampling percentages should be equal to the number of bumps defined by the relevance threshold. "
                f"Got {self.getSamplingStrategy()} instead."
            )

        if any(sampling_percentage <= 0 for sampling_percentage in self.getSamplingStrategy()) or all(
                sampling_percentage <= 1 for sampling_percentage in self.getSamplingStrategy()) or all(
            sampling_percentage >= 1 for sampling_percentage in self.getSamplingStrategy()):
            raise ValueError(
                f"When 'sampling_strategy' is a list, all the sampling percentages should be greater than 0, at least one sampling percentage should be greater than 1 and at least one sampling percentage should be less than 1. "
                f"Got {self.getSamplingStrategy()} instead."
            )

        for bump, sampling_percentage in zip(bumps, self.getSamplingStrategy()):
            bump.sampling_percentage = sampling_percentage

        return bumps

    def _validate_sampling_strategy_balance(self, bumps):
        target_bump_size = sum([bump.size for bump in bumps]) / len(bumps)

        for bump in bumps:
            bump.sampling_percentage = target_bump_size / bump.size

        return bumps

    def _validate_sampling_strategy_extreme(self, bumps):
        n_samples = sum([bump.size for bump in bumps])
        n_bumps = len(bumps)

        avg_bump_size = n_samples / n_bumps

        scale = n_samples / sum([avg_bump_size ** 2 / bump.size for bump in bumps])

        target_bump_sizes = [avg_bump_size ** 2 / bump.size * scale for bump in bumps]

        for bump, target_bump_size in zip(bumps, target_bump_sizes):
            bump.sampling_percentage = target_bump_size / bump.size

        return bumps

    @abstractmethod
    def _oversample(self, bump):
        raise NotImplementedError()

    @abstractmethod
    def _undersample(self, bump):
        raise NotImplementedError()

    def _resample(self, bump):
        if bump.sampling_percentage == 1:
            return bump.samples
        elif bump.sampling_percentage > 1:
            return self._oversample(bump)
        else:
            return self._undersample(bump)
