from pyspark.ml.param import Param, Params, TypeConverters

from src.params.relevance._phi import _PhiParams


class _RelevanceParams(_PhiParams):
    threshold = Param(
        Params._dummy(),
        "threshold",
        "The relevance threshold. ((0, 1))",
        typeConverter=TypeConverters.toFloat
    )

    def __init__(self):
        super(_RelevanceParams, self).__init__()
        self._setDefault(threshold=0.8)

    def getThreshold(self):
        return self.getOrDefault(self.threshold)

    def setThreshold(self, value):
        return self._set(threshold=value)
