import os

from chemicalchecker.util import logged
from .evaluation import Validation
from .multi import MultiSetup



@logged
class MultiValidation(MultiSetup):
    """Helper class to create multiple validations."""
    def __init__(self,
                 data,
                 models_path,
                 overwrite,
                 wipe=True,
                 # only_validation=True,
                 # only_train=False,
                 **kwargs):
        """Initialize multi-validation.

        Args:
            data(list): List of data files.
            models_path(list): Parent directory where to store the results.
            datasets(list or list of lists): Datasets to be used.
            overwrite(bool): Overwrite results. Otherwise, checks if a validation.txt file has been written.
            wipe(bool): Initiate wiping of files after TM run (default = True).
            only_validation(bool): Keep only validation data when wiping files (default = True).
            only_train(bool): Use all data as training data, no test or validation carried out (default = False).
            **kwargs: Arguments of the Validation class.
        """
        self.__log.info("Initialize multiple validation")
        self.overwrite = overwrite
        MultiSetup.__init__(self, data=data, models_path=models_path, **kwargs)
        self.validation = Validation(**kwargs)
        self.wipe=wipe
        # self.only_validation=only_validation
        # self.only_train=only_train
        #
        # self.model_type = model_type

    def _is_done(self, models_path):
        if os.path.exists(os.path.join(models_path, "validation.txt")):
            self.__log.info("Validation exists in %s" % models_path)
            return True
        elif os.path.exists(os.path.join(models_path, "bases")):
            for mod in os.listdir(os.path.join(models_path, "bases")):
                if mod.endswith((".z")):
                    self.__log.info("Only train model exists in %s" % models_path)
                    return True
            return False
        else:
            return False
    def _get_signature_type(self, signature_type, parameters):
        if signature_type == 'CC':
            parameters['use_cc'] = True
        elif signature_type == 'Signaturizer':
            parameters['use_cc'] = False
            parameters['is_classic'] = False
        elif signature_type == 'ECFP4':
            parameters['use_cc'] = False
            parameters['is_classic'] = False
        elif type(signature_type) == str:
            self.__log.info("Signature type not available")
            return None
        else:
            self.__log.info("Setting signature type from variables")
        return parameters

    def run(self, TargetMate, use_inchikey, signature_type = None, scramble=False, set_train_idx= None, set_test_idx = None, **kwargs):
        """ Create instances of TargetMate

        Args:
            TargetMate (object): TargetMate instance which specifies the type of model which should be created
            use_inchikey (bool): Use inchikeys to create signatures
            scramble (bool): Scramble y values when creating/validating models to create a random comparison
            set_train_idx (list): Preselected train idxs # TODO: Shouldn't this be within individual instances of TM rather than the wrapper
            set_test_idx (list): Preselected test idxs
            **
            wa: rgs:
        """

        sign_paths = None # Added by Paula: currently not using this so just going to set to None 17/04/21
        kwargs = self._get_signature_type(signature_type, kwargs)
        if kwargs is None:
            return None

        self.__log.info("Multiple trainings")
        tm_list   = []
        data_list = []
        i=0
        for data, models_path, _ in self.tasks:
            if not self.overwrite:
                if self._is_done(models_path):
                    continue
            tm = TargetMate(models_path=models_path, use_inchikey=use_inchikey,
                            master_sign_paths=sign_paths,
                            is_tmp_bases = self.validation.is_tmp_bases,
                            is_tmp_signatures=self.validation.is_tmp_signatures,
                            is_tmp_predictions=self.validation.is_tmp_predictions,
                            **kwargs)
            dat = tm.get_data_fit(data, use_inchikey=use_inchikey, **kwargs) # Edited by Paula: inchikey_idx was set to -1, removed so that user must input it
            if dat is not None:

                tm_list += [tm.on_disk()]
                data_list += [dat.on_disk(tm.tmp_path)]
                i += 1

        if len(tm_list) > 0:
            self.validation.validate(tm=tm_list, data=data_list, wipe=self.wipe,
                                     scramble = scramble,
                                     set_train_idx = set_train_idx, set_test_idx= set_test_idx)


class CompleteModel:

    def __init__(self,
                 directories,
                 models_path=None,
                 **kwargs):
        """
        Helper class to carry out train/test split, followed by model using all data to train.

        Args:
            directories(list): List of directories in which to store models [validation, full model]
            **kwargs: arguments of the MultiValidation class
        """

        assert models_path is None, "If using CompleteModel please enter directories not models path."
        assert len(directories) == 2, "Please enter directories to store models."
        self.directories = directories
        self.kwargs = kwargs

    def run_full(self, **kwargs):
        mv = MultiValidation(models_path=self.directories[0], model_type = 'val', **self.kwargs)
        mv.run(**kwargs)
        mv_full = MultiValidation(models_path=self.directories[1], model_type = 'only_train', **self.kwargs)
        mv_full.run(**kwargs)

class MultiFit(MultiSetup):
    #TODO
    pass


class MultiPredict(MultiSetup):
    #TODO
    pass
