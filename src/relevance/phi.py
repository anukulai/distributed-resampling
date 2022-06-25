import pandas as pd
import smogn
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import DoubleType

from src.params._common import _InputColParam, _OutputColParam
from src.params.relevance._phi import _PhiParams


class Phi(Transformer, _InputColParam, _OutputColParam, _PhiParams):
    @keyword_only
    def __init__(self, input_col=None, output_col=None, method="auto", xtrm_type="both", coef=1.5,
                 ctrl_pts_region=None):
        super(Phi, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, input_col=None, output_col=None, method="auto", xtrm_type="both", coef=1.5,
                  ctrl_pts_region=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        input_col = self.getInputCol()
        output_col = self.getOutputCol()

        method = self.getMethod()
        xtrm_type = self.getXtrmType()
        coef = self.getCoef()
        ctrl_pts_region = self.getCtrlPtsRegion()

        @pandas_udf(DoubleType())
        def to_phi(y: pd.Series) -> pd.Series:
            ctrl_pts = smogn.phi_ctrl_pts(
                y=y,
                method=method,
                xtrm_type=xtrm_type,
                coef=coef,
                ctrl_pts=ctrl_pts_region
            )

            y_phi = smogn.phi(
                y=y,
                ctrl_pts=ctrl_pts
            )

            y_phi = pd.Series(y_phi)

            return y_phi

        return dataset.withColumn(output_col, to_phi(dataset[input_col]))
