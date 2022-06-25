class Bump:
    def __init__(self, type, samples):
        self.type = type
        self.samples = samples
        self.size = samples.count()
