from pyspark.ml.param import Param, Params, TypeConverters


class _PhiParams(Params):
    method = Param(
        Params._dummy(),
        "method",
        "The relevance method. ('auto', 'manual')",
        typeConverter=TypeConverters.toString
    )

    xtrm_type = Param(
        Params._dummy(),
        "xtrm_type",
        "The distribution focus. ('high', 'low', 'both')",
        typeConverter=TypeConverters.toString
    )

    coef = Param(
        Params._dummy(),
        "coef",
        "The box plot coefficient. (>= 0)",
        typeConverter=TypeConverters.toFloat
    )

    ctrl_pts_region = Param(
        Params._dummy(),
        "ctrl_pts_region",
        "The regions of interest for manual relevance method.",
        typeConverter=TypeConverters.toListListFloat
    )

    def __init__(self):
        super(_PhiParams, self).__init__()
        self._setDefault(method="auto", xtrm_type="both", coef=1.5, ctrl_pts_region=None)

    def getMethod(self):
        return self.getOrDefault(self.method)

    def setMethod(self, value):
        return self._set(method=value)

    def getXtrmType(self):
        return self.getOrDefault(self.xtrm_type)

    def setXtrmType(self, value):
        return self._set(xtrm_type=value)

    def getCoef(self):
        return self.getOrDefault(self.coef)

    def setCoef(self, value):
        return self._set(coef=value)

    def getCtrlPtsRegion(self):
        return self.getOrDefault(self.ctrl_pts_region)

    def setCtrlPtsRegion(self, value):
        return self._set(ctrl_pts_region=value)
