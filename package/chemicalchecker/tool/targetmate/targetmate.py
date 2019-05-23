
from sklearn.model_selection import check_cv

@logged
class TargetMate:
	"""TargetMate Classifier

    Parameters
    ----------

    cv : int or cross-validation generator, optional (default=None)
        The default cross-validation generator used is Stratified K-Folds.
        If an integer is provided, then it is the number of folds used.
        See the module :mod:`sklearn.model_selection` module for the
        list of possible cross-validation objects.


	"""
    def __init__(self, base_clf = None, cv = 5, datasets = "all", models_path = None):
    	# Set the base classifier
        if not base_clf:
        	from sklearn.linear_model import LogisticRegressionCV
        	self.base_clf = LogisticRegressionCV(class_weight = "balanced")
        else:
        	self.base_clf = base_clf
        # Crossvalidation to determine the performances of the individual predictors
        self.cv = cv
        # Store the paths to the sign3 (only the ones that have been already trained)
        if datasets not datasets:
            self.datasets = ["%s%d.001" for x in ["A","B","C","D","E"] for i in [1,2,3,4,5]] # TO-DO: @Martino, is there a way to just check how many signatures are available?
        else:
            self.datasets = datasets
        
        # XXX
        if models_path is None
            self.models_path = ""
        else:
        	self.models_path = models_path
        # XXX
        self.datasets
        
	def fit(self, data, standardise = True):
        """
        XXXX
        """
        # Read data if it is a file
        if type(data) == str:
            with open(data, "r") as f:
                data = []
                for r in csv.reader(f, delimiter = "\t"):
                	data += [(r[0], int(r[1]))]
        # Get signatures
        self.__log.info("Calculating sign3 for every molecule. Invalid SMILES will be removed.")
        

        # Initialize cross-validation generator
        cv = check_cv(self.cv, y, classifier=True)
        folds = list(cv.split(X, y))

            
    def predict(self, data_path):
        ""


	def predict_proba(self, ):
        ""
