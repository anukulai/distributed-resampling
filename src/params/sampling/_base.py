from pyspark.ml.param import Param, Params, TypeConverters

from src.params._common import _KPartitionsParam
from src.params.relevance._relevance import _RelevanceParams


class _BaseSamplerParams(_KPartitionsParam, _RelevanceParams):
    sampling_strategy = Param(
        Params._dummy(),
        "sampling_strategy",
        "The sampling strategy.",
        typeConverter=TypeConverters.identity
    )

    def __init__(self):
        super(_BaseSamplerParams, self).__init__()
        self._setDefault(sampling_strategy="balance")

    def getSamplingStrategy(self):
        return self.getOrDefault(self.sampling_strategy)

    def setSamplingStrategy(self, value):
        return self._set(sampling_strategy=value)
