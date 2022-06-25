from pyspark.ml.param import Param, Params, TypeConverters


class _KMeansParams(Params):
    init_steps = Param(
        Params._dummy(),
        "init_steps",
        "The number of initialization steps. (> 0)",
        typeConverter=TypeConverters.toInt
    )

    tol = Param(
        Params._dummy(),
        "tol",
        "The convergence tolerance. (>= 0)",
        typeConverter=TypeConverters.toFloat
    )

    max_iter = Param(
        Params._dummy(),
        "max_iter",
        "The maximum number of iterations. (>= 0)",
        typeConverter=TypeConverters.toInt
    )

    def __init__(self):
        super(_KMeansParams, self).__init__()
        self._setDefault(init_steps=2, tol=1e-4, max_iter=20)

    def getInitSteps(self):
        return self.getOrDefault(self.init_steps)

    def setInitSteps(self, value):
        return self._set(init_steps=value)

    def getTol(self):
        return self.getOrDefault(self.tol)

    def setTol(self, value):
        return self._set(tol=value)

    def getMaxIter(self):
        return self.getOrDefault(self.max_iter)

    def setMaxIter(self, value):
        return self._set(max_iter=value)


class _SMOGNParams(Params):
    k_neighbours = Param(
        Params._dummy(),
        "k_neighbours",
        "The number of nearest neighbours.",
        typeConverter=TypeConverters.toInt
    )

    perturbation = Param(
        Params._dummy(),
        "perturbation",
        "The perturbation.",
        typeConverter=TypeConverters.toFloat
    )

    def __init__(self):
        super(_SMOGNParams, self).__init__()
        self._setDefault(k_neighbours=5, perturbation=0.02)

    def getKNeighbours(self):
        return self.getOrDefault(self.k_neighbours)

    def setKNeighbours(self, value):
        return self._set(k_neighbours=value)

    def getPerturbation(self):
        return self.getOrDefault(self.perturbation)

    def setPerturbation(self, value):
        return self._set(perturbation=value)
