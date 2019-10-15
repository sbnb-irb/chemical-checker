import pandas as pd

@logged
class InputData:
    """A simple input data class"""
    
    def __init__(self, data):
        """Initialize input data class"""
        self.idx       = np.array([d[0] for d in data])
        self.activity  = np.array([float(d[1]) for d in data])
        self.smiles    = np.array([d[2] for d in data])
        self.inchikey  = np.array([d[3] for d in data])

    def __iter__(self):
        for idx, v in self.as_dataframe().iterrows():
            yield v

    def as_dataframe(self):
        df = pd.DataFrame({
            "idx": self.idx,
            "activity": self.activity,
            "smiles": self.smiles,
            "inchikey": self.inchikey
            })
        return df        

    def shuffle(self):
        """Shuffle data"""
        ridxs = [i for i in range(0, len(self.idx))]
        random.shuffle(ridxs)
        self.idx      = self.idx[ridxs]
        self.activity = self.activity[ridxs]
        self.smiles   = self.smiles[ridxs]
        self.inchikey = self.inchikey[ridxs]


@logged
class OutputData:
    """A simple output data class"""

    def __init__(self, data):
        """Initialize output data class"""
        pass