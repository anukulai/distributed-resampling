from src.relevance.phi import Phi
from src.utils.dataframe import collect_col


def collect_phi(dataset, label_col, method="auto", xtrm_type="both", coef=1.5, ctrl_pts_region=None):
    relevance_col = "phi"
    dataset = Phi(input_col=label_col, output_col=relevance_col, method=method, xtrm_type=xtrm_type, coef=coef,
                  ctrl_pts_region=ctrl_pts_region).transform(dataset)

    return collect_col(dataset, relevance_col)
