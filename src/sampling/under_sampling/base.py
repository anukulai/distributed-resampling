from abc import abstractmethod

from src.sampling.base import BaseSampler


class BaseUnderSampler(BaseSampler):
    def _validate_sampling_strategy_number(self, bumps):
        if self.getSamplingStrategy() <= 0 or self.getSamplingStrategy() >= 1:
            raise ValueError(
                f"When 'sampling_strategy' is a number, it should be greater than 0 and less than 1. "
                f"Got {self.getSamplingStrategy()} instead."
            )

        for bump in bumps:
            if bump.type == "normal":
                bump.sampling_percentage = self.getSamplingStrategy()
            else:
                bump.sampling_percentage = 1

        return bumps

    def _validate_sampling_strategy_list(self, bumps):
        if len(self.getSamplingStrategy()) != len([bump for bump in bumps if bump.type == "normal"]):
            raise ValueError(
                f"When 'sampling_strategy' is a list, the number of sampling percentages should be equal to the number of normal bumps defined above the relevance threshold. "
                f"Got {self.getSamplingStrategy()} instead."
            )

        if any(sampling_percentage <= 0 or sampling_percentage > 1 for sampling_percentage in
               self.getSamplingStrategy()) or all(
            sampling_percentage == 1 for sampling_percentage in self.getSamplingStrategy()):
            raise ValueError(
                f"When 'sampling_strategy' is a list, all the sampling percentages should be greater than 0 and less than or equal to 1 and at least one sampling percentage should not be equal to 1. "
                f"Got {self.getSamplingStrategy()} instead."
            )

        for bump in bumps:
            if bump.type == "normal":
                bump.sampling_percentage = self.getSamplingStrategy().pop(0)
            else:
                bump.sampling_percentage = 1

        return bumps

    def _validate_sampling_strategy_balance(self, bumps):
        target_bump_size = min([bump.size for bump in bumps])

        for bump in bumps:
            bump.sampling_percentage = target_bump_size / bump.size

        return bumps

    def _validate_sampling_strategy_extreme(self, bumps):
        # TODO
        pass

    @abstractmethod
    def _undersample(self, bump):
        raise NotImplementedError()

    def _resample(self, bump):
        if bump.sampling_percentage == 1:
            return bump.samples
        else:
            return self._undersample(bump)
