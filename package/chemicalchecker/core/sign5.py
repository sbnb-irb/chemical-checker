"""Signature 2 based on Siamese networks"""


from .signature_base import BaseSignature
from .signature_data import DataSignature

from chemicalchecker.util import logged

DEFAULT_T = 0.01

@logged
class sign5(BaseSignature, DataSignature):

    def __init__(self, signature_path, dataset, **params):
        """Initialize the signature.

        Args:
            signature_path(str): the path to the signature directory.
            model_path(str): Where the persistent model is.
        """
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(
            self, signature_path, dataset, **params)
        self.__log.debug('signature path is: %s', signature_path)
        self.data_path = os.path.join(self.signature_path, "sign2.h5")
        DataSignature.__init__(self, self.data_path)

    def copy_sign1_to_sign2(self, s1, s2, just_data=False):
        """Copy from sign0 to sign1"""
        if s1.molset != s2.molset:
            raise Exception(
                "Copying from signature 1 to 2 is only allowed for same molsets (reference or full)")
        self.__log.debug("Copying HDF5 dataset")
        with h5py.File(s2.data_path, "w") as hf:
            hf.create_dataset(
                "name", data=np.array([str(self.dataset) + "sig"], DataSignature.string_dtype()))
            hf.create_dataset(
                "date", data=np.array([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")], DataSignature.string_dtype()))
            hf.create_dataset("V", data=s1[:])
            hf.create_dataset("keys", data=np.array(
                s1.keys, DataSignature.string_dtype()))
            if s1.molset == "reference":
                mappings = s1.get_h5_dataset("mappings")
                hf.create_dataset("mappings", data=np.array(
                    mappings, DataSignature.string_dtype()))
        if not just_data:
            self.__log.debug("Copying triplets")
            fn1 = os.path.join(s1.model_path, "triplets.h5")
            if os.path.exists(fn1):
                self.__log.debug("Triplets are available.")
                fn2 = os.path.join(s2.model_path, "triplets.h5")
                shutil.copyfile(fn1, fn2)
            else:
                self.__log.warn(
                    "No triplets available! Please fit sign0 with option do_triplets=True")
        self.refresh()
        s1.refresh()
        s2.refresh()

    def get_self_triplets(self, local_neig_path, num_triplets=1000000):
        """Get triplets of signatures only looking at itself"""
        s2 = self.get_molset("reference")
        if local_neig_path:
            neig_path = os.path.join(s2.model_path, "neig.h5")
        else:
            neig_path = s2.get_neighbors().data_path
        opt_t = DEFAULT_T
        # Heuristic to correct opt_t, dependent on the size of the data
        LB = 10000
        UB = 100000
        TMAX = 50
        TMIN = 10
        def get_t_max(n):
            n = np.clip(n, LB, UB)
            a = (TMAX-TMIN)/(LB-UB)
            b = TMIN - a*UB
            return a*n+b
        with h5py.File(neig_path, "r") as hf:
            N, kn = hf["indices"].shape
            opt_t = np.min([opt_t, 0.01])
            k = np.clip(opt_t * N, 5, 100)
            k = np.min([k, kn * 0.5 + 1])
            k = np.max([k, 5])
            k = np.min([k, get_t_max(N)])
            k = int(k)
            self.__log.debug("... selected T is %d" % k)
            nn_pos = hf["indices"][:, 1:(k + 1)]
            nn_neg = hf["indices"][:, (k + 1):]
        self.__log.debug("Starting sampling (pos:%d, neg:%d)" %
                         (nn_pos.shape[1], nn_neg.shape[1]))
        n_sample = min(int(num_triplets / N), 100)
        triplets = []
        med_neg = nn_neg.shape[1]
        nn_pos_prob = [(len(nn_pos) - i) for i in range(0, nn_pos.shape[1])]
        nn_neg_prob = [(len(nn_neg) - i) for i in range(0, nn_neg.shape[1])]
        nn_pos_prob = np.array(nn_pos_prob) / np.sum(nn_pos_prob)
        nn_neg_prob = np.array(nn_neg_prob) / np.sum(nn_neg_prob)
        for i in range(0, N):
            # sample positives with replacement
            pos = np.random.choice(
                nn_pos[i], n_sample, p=nn_pos_prob, replace=True)
            if n_sample > med_neg:
                # sample "medium" negatives
                neg = np.random.choice(
                    nn_neg[i], med_neg, p=nn_neg_prob, replace=False)
                # for the rest, sample "easy" negatives
                forb = set(list(nn_pos[i]) + list(nn_neg[i]))
                cand = [i for i in range(0, N) if i not in forb]
                if len(cand) > 0:
                    neg_ = np.random.choice(
                        cand, min(len(cand), n_sample - med_neg), replace=True)
                    neg = np.array(list(neg) + list(neg_))
            else:
                neg = np.random.choice(nn_neg[i], n_sample, replace=False)
            if len(pos) > len(neg):
                neg = np.random.choice(neg, len(pos), replace=True)
            elif len(pos) < len(neg):
                neg = np.random.choice(neg, len(pos), replace=False)
            else:
                pass
            for p, n in zip(pos, neg):
                triplets += [(i, p, n)]
        triplets = np.array(triplets).astype(np.int)
        fn = os.path.join(s2.model_path, "triplets_self.h5")
        self.__log.debug("Triplets path: %s" % fn)
        with h5py.File(fn, "w") as hf:
            hf.create_dataset("triplets", data=triplets)
        return triplets

    def parametrize(self, dim)

        if dim > 1024:
            layers_sizes = [512, 128]
        elif dim > 512:
            layers_sizes = [256, 128]
        elif dim 

        params = {
            'epochs': 3,
            'dropouts': [None],
            'layers_sizes': None,
            'learning_rate': "auto",
            'batch_size': 128,
            'activations': ["tanh"],
            'layer': [Dense],
            'loss_func': 'orthogonal_tloss',
            'margin': 1.0,
            'alpha': 1.0
        }

        return params


    def fit(self, sign1):
        pass

    def predict(self, sign1):
        pass


