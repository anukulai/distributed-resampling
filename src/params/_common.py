from pyspark.ml.param import Param, Params, TypeConverters


class _InputColParam(Params):
    input_col = Param(
        Params._dummy(),
        "input_col",
        "The input column name.",
        typeConverter=TypeConverters.toString
    )

    def __init__(self):
        super(_InputColParam, self).__init__()
        self._setDefault(input_col=None)

    def getInputCol(self):
        return self.getOrDefault(self.input_col)

    def setInputCol(self, value):
        return self._set(input_col=value)


class _OutputColParam(Params):
    output_col = Param(
        Params._dummy(),
        "output_col",
        "The output column name.",
        typeConverter=TypeConverters.toString
    )

    def __init__(self):
        super(_OutputColParam, self).__init__()
        self._setDefault(output_col=None)

    def getOutputCol(self):
        return self.getOrDefault(self.output_col)

    def setOutputCol(self, value):
        return self._set(output_col=value)


class _LabelColParam(Params):
    label_col = Param(
        Params._dummy(),
        "label_col",
        "The label column name.",
        typeConverter=TypeConverters.toString
    )

    def __init__(self):
        super(_LabelColParam, self).__init__()
        self._setDefault(label_col=None)

    def getLabelCol(self):
        return self.getOrDefault(self.label_col)

    def setLabelCol(self, value):
        return self._set(label_col=value)


class _KPartitionsParam(Params):
    k_partitions = Param(
        Params._dummy(),
        "k_partitions",
        "The number of partitions. (> 1)",
        typeConverter=TypeConverters.toInt
    )

    def __init__(self):
        super(_KPartitionsParam, self).__init__()
        self._setDefault(k_partitions=2)

    def getKPartitions(self):
        return self.getOrDefault(self.k_partitions)

    def setKPartitions(self, value):
        return self._set(k_partitions=value)
