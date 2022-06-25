from pyspark import keyword_only

from src.sampling.over_sampling.base import BaseOverSampler


class DistributedROS(BaseOverSampler):
    @keyword_only
    def __init__(self, label_col=None, sampling_strategy="balance", k_partitions=2, threshold=0.8, method="auto",
                 xtrm_type="both", coef=1.5, ctrl_pts_region=None):
        super(DistributedROS, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, label_col=None, sampling_strategy="balance", k_partitions=2, threshold=0.8, method="auto",
                  xtrm_type="both", coef=1.5, ctrl_pts_region=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _oversample(self, bump):
        return self._partition(bump.samples).sample(withReplacement=True, fraction=bump.sampling_percentage)
