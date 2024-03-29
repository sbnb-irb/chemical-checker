"""Molrepo definition.

The Molrepo is a molecule repository (aka library or collection) exposing
mappings between various textual representations (SMILES, InChI and InChIKey)
for different set of molecules.

Example::

    from chemicalchecker.database import Molrepo
    molrep = Molrepo.get('drugbank')[0]
    len(molrep.molecules)
    >>> 9167

"""
import os
import datetime
import tempfile
from time import time
import sqlalchemy
from sqlalchemy.dialects import postgresql
from sqlalchemy import Column, Text, Boolean, ForeignKey, VARCHAR
from sqlalchemy.orm import class_mapper, ColumnProperty, relationship

from .molecule import Molecule
from .database import Base, get_engine, get_session

from chemicalchecker.util.hpc import HPC
from chemicalchecker.util.parser import Parser
from chemicalchecker.util import logged, Config
from chemicalchecker.util.decorator import cached_property


@logged
class Molrepo(Base):
    """Molrepo table class.

    This table offer a mapping between inchikeys and different external
    compound ids (e.g. chembl, bindigdb, etc.).

    Fields:
        id(str): primary key, src_id + "_" + molrepo_name.
        molrepo_name(str): the molrepo name.
        src_id(str): the download id as in the source file.
        smiles(str): simplified molecular-input line-entry system (SMILES).
        inchikey(bool): hashed version of the full InChI (SHA-256 algorithm).
        inchi(bool): International Chemical Identifier (InChI).
    """

    __tablename__ = 'molrepo'
    molrepo_name = Column(Text, primary_key=True)
    description = Column(Text)
    universe = Column(Boolean)
    essential = Column(Boolean)

    datasources = relationship("Datasource",
                               secondary="molrepo_has_datasource",
                               back_populates="molrepos",
                               lazy='joined')

    def __repr__(self):
        """String representation."""
        return str(self.molrepo_name)

    @staticmethod
    def _create_table():
        engine = get_engine()
        Base.metadata.create_all(engine, tables=[Molrepo.__table__])

    @staticmethod
    def _drop_table():
        engine = get_engine()
        Molrepo.__table__.drop(engine)

    @staticmethod
    def _table_exists():
        engine = get_engine()
        return sqlalchemy.inspect(engine).has_table(Molrepo.__tablename__)

    @staticmethod
    def _table_attributes():
        attrs = [a for a in class_mapper(Molrepo).iterate_properties]
        col_attrs = [a.key for a in attrs if isinstance(a, ColumnProperty)]
        input_attrs = [a for a in col_attrs if a != 'id']
        return input_attrs

    @staticmethod
    def add(kwargs):
        """Add a new row to the table.

        Args:
            kwargs(dict):The data in dictionary format.
        """
        if type(kwargs) is dict:
            molrepo = Molrepo(**kwargs)
        else:
            raise Exception("Input data for add method is not a dictionary")
        Molrepo.__log.debug(molrepo)
        session = get_session()
        session.add(molrepo)
        session.commit()
        session.close()

    @staticmethod
    def get(name=None):
        """Get molrepos associated to the given name.

        Args:
            name(str):The molrepo name, e.g "chebi"
        """
        params = {}
        if name is not None:
            params["molrepo_name"] = name

        session = get_session()

        if len(params) == 0:
            query = session.query(Molrepo)
        else:
            query = session.query(Molrepo).filter_by(**params)

        res = query.all()
        session.close()
        return res

    @cached_property
    def molecules(self):
        """Fetch molecules for Molrepo."""
        params = {}
        params["molrepo_name"] = self.molrepo_name
        session = get_session()
        query = session.query(MolrepoHasMolecule).filter_by(**params)
        res = query.all()
        session.close()
        return res

    @staticmethod
    def to_csv(staticmethod, filename):
        """Write molecules InChI-Key, source_id, InChI and SMILES to CSV file.

        Args:
            filename(str): Path to a CSV file.
        """
        import pandas as pd
        molecules = Molrepo.get_by_molrepo_name(molrepo_name)
        df = pd.DataFrame(molecules, columns=['molrepo', 'source_id',
                                              'SMILES', 'InChIKey', 'InChI'])
        df.dropna(inplace=True)
        df.sort_values('InChIKey', inplace=True)
        df[['InChIKey', 'source_id', 'SMILES', 'InChI']].to_csv(
            filename, index=False)

    @staticmethod
    def from_csv(filename):
        """Add entries from CSV file.

        Args:
            filename(str): Path to a CSV file.
        """
        import pandas as pd
        df = pd.read_csv(filename)
        # The boolean columns must be changed to boolean values otherwise
        # SQLalchmy passes strings
        df.universe = df.universe.apply(lambda x: False if x == 'f' else True)
        df.essential = df.essential.apply(
            lambda x: False if x == 'f' else True)

        # check columns
        needed_cols = Molrepo._table_attributes()
        if needed_cols != list(df.columns):
            raise Exception("Input missing columns: %s", ' '.join(needed_cols))
        # add them
        for row_nr, row in df.iterrows():
            try:
                Molrepo.add(row.dropna().to_dict())
            except Exception as err:
                Molrepo.__log.error(
                    "Error in line %s: %s", row_nr, str(err))

    @staticmethod
    def get_universe_molrepos():
        """Get Molrepo names that are considered universe."""
        session = get_session()

        query = session.query(Molrepo.molrepo_name).filter(
            (Molrepo.universe)).distinct(
            Molrepo.molrepo_name)
        res = query.all()
        session.close()
        return res

    @staticmethod
    def get_by_molrepo_name(molrepo_name, only_raw=False):
        """Get Molrepo entries associated to the given name.

        Args:
            molrepo_name(str): The molrepo_name to search for.
            only_raw(bool): Only get the raw values without the whole object
                (default:false)
        """
        session = get_session()
        query = session.query(
            MolrepoHasMolecule.molrepo_name,
            MolrepoHasMolecule.src_id,
            MolrepoHasMolecule.smiles,
            MolrepoHasMolecule.inchikey,
            Molecule.inchi
        ).outerjoin(
            Molecule,
            Molecule.inchikey == MolrepoHasMolecule.inchikey
        ).filter(
            MolrepoHasMolecule.molrepo_name == molrepo_name)
        if only_raw:
            res = query.with_entities(
                MolrepoHasMolecule.molrepo_name,
                MolrepoHasMolecule.src_id,
                MolrepoHasMolecule.smiles,
                MolrepoHasMolecule.inchikey,
                Molecule.inchi).all()
        else:
            res = query.all()
        session.close()
        return res

    @staticmethod
    def get_fields_by_molrepo_name(molrepo_name, fields=None):
        """Get specified column fields.

        Get specified column fields from a molrepo_name in raw format (tuples)

        Args:
            molrepo_name(str): The molrepo_name to search for.
            fields(list): List of field names. If None, all fields.
        """
        if fields is None:
            return Molrepo.get_by_molrepo_name(molrepo_name, True)

        cols = MolrepoHasMolecule._table_attributes()
        query_fields = []

        for field in fields:
            if field in cols or field == "inchi":
                if field == "inchi":
                    query_fields.append("Molecule." + field)
                else:
                    query_fields.append("MolrepoHasMolecule." + field)

        if len(query_fields) == 0:
            return None

        session = get_session()
        query = session.query(MolrepoHasMolecule).outerjoin(
            Molecule, Molecule.inchikey == MolrepoHasMolecule.inchikey).filter(
            MolrepoHasMolecule.molrepo_name ==
            molrepo_name, MolrepoHasMolecule.inchikey.isnot(None))
        res = query.with_entities(*[eval(f) for f in query_fields]).all()

        session.close()
        return res

    @staticmethod
    def count(molrepo_name=None):
        """Get Molrepo entries associated to the given source name.

        Args:
            molrepo_name(str): The source name from `Datasource.molrepo_name`
        """
        session = get_session()
        if molrepo_name:
            query = session.query(MolrepoHasMolecule).filter_by(
                molrepo_name=molrepo_name).count()
        else:
            query = session.query(MolrepoHasMolecule).count()
        return int(query)

    @staticmethod
    def from_molrepo_name(molrepo_name):
        """Fill Molrepo table from a molrepo name.

        Args:
            molrepo_name(str): a molrepo name.
        """
        molrepo = Molrepo.get(molrepo_name)
        if len(molrepo) == 0:
            raise Exception(
                "Molrepo name %s file not available.", molrepo_name)

        map_files = {}

        for ds in molrepo[0].datasources:
            path = ds.data_path
            if ds.filename is not None and ds.is_db is False:
                path = os.path.join(path, ds.filename)
            map_files[ds.datasource_name] = path
            #Molrepo._log.debug("Importing Datasource %s", ds.datasource_name)
            ds.download()
        molrepo_parser = molrepo_name

        # parser_fn yield a list of dictionaries with keys as a molrepo entry
        parse_fn = Parser.parse_fn(molrepo_parser)
        # profile time
        t_start = time()
        engine = get_engine(  ) 
        with engine.begin() as conn:
            for chunk in parse_fn(map_files, molrepo_name, 1000):
                if len(chunk) == 0:
                    continue
                chunk_inchi = []
                chunk_molrepo = []
                for data in chunk:
                    if data["inchikey"] is not None:
                        chunk_inchi.append({"inchikey": data["inchikey"],
                                            "inchi": data["inchi"]})
                    del data["inchi"]
                    chunk_molrepo.append(data)
                if len(chunk_inchi) > 0:
                    conn.execute(postgresql.insert(
                        Molecule.__table__).values(
                        chunk_inchi).on_conflict_do_nothing(
                        index_elements=[Molecule.inchikey]))
                conn.execute(postgresql.insert(
                    MolrepoHasMolecule.__table__).values(
                    chunk_molrepo).on_conflict_do_nothing(
                    index_elements=[MolrepoHasMolecule.id]))
        t_end = time()
        t_delta = str(datetime.timedelta(seconds=t_end - t_start))
        Molrepo.__log.info(
            "Importing Molrepo Name %s took %s", molrepo_name, t_delta)

    @staticmethod
    def molrepo_sequential(tmpdir, only_essential=False, **kwargs):
        #Molrepo._log.info("Generating mol repositories - sequential" )
        
        molrepos_names = set()
        molrepos = Molrepo.get()
        for molrepo in molrepos:
            if only_essential and not molrepo.essential:
                continue
            molrepos_names.add(molrepo.molrepo_name)
        
        for ds in molrepos_names:
            Molrepo.from_molrepo_name(ds)
    
    @staticmethod
    def molrepo_hpc(tmpdir, only_essential=False, **kwargs):
        """Run HPC jobs importing all molrepos.

        tmpdir(str): Folder (usually in scratch) where the job directory is
            generated.
        only_essential(bool): Only the essentail molrepos (default:false)
        """
        cc_config = kwargs.get("cc_config", os.environ['CC_CONFIG'])
        cfg = Config(cc_config)
        job_path = tempfile.mkdtemp(prefix='jobs_molrepos_', dir=tmpdir)
        # create job directory if not available
        if not os.path.isdir(job_path):
            os.mkdir(job_path)
        # create script file
        script_lines = [
            "import sys, os",
            "import pickle",
            "from chemicalchecker.database import Datasource",
            "from chemicalchecker.database import Molrepo",
            "task_id = sys.argv[1]",  # <TASK_ID>
            "filename = sys.argv[2]",  # <FILE>
            "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
            "data = inputs[task_id]",  # elements for current job
            "for d in data:",  # elements are indexes
            "    Molrepo.from_molrepo_name(d)",  # start import
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'molrepo_script.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        molrepos_names = set()
        molrepos = Molrepo.get()
        for molrepo in molrepos:
            if only_essential and not molrepo.essential:
                continue
            molrepos_names.add(molrepo.molrepo_name)

        params = {}
        params["num_jobs"] = len(molrepos_names)
        params["jobdir"] = job_path
        params["job_name"] = "CC_MOLREPO"
        params["elements"] = list(molrepos_names)
        params["wait"] = True
        params["check_error"] = False
        params["memory"] = 16
        # job command
        singularity_image = cfg.PATH.SINGULARITY_IMAGE
        command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={}" \
            " singularity exec {} python {} <TASK_ID> <FILE>"
        command = command.format(
            os.path.join(cfg.PATH.CC_REPO, 'package'), cc_config,
            singularity_image, script_name)
        # submit jobs
        cluster = HPC.from_config(cfg)
        cluster.submitMultiJob(command, **params)
        return cluster


@logged
class MolrepoHasMolecule(Base):
    """Molrepo-Molecule association object.

    Again a Many-to-Many relationship.
    This table links Molecules and Molrepos also including the external
    compound identifiers (e.g. ChEMBL -> ``CHEMBL10``,
    BindigDB -> ``BDBM50028883``, etc.).

    Fields:
        id(str): primary key, src_id + "_" + molrepo_name.
        molrepo_name(str): the molrepo name.
        src_id(str): the download id as in the source file.
        smiles(str): simplified molecular-input line-entry system (SMILES).
        inchikey(str): hashed version of the full InChI (SHA-256 algorithm).
    """

    __tablename__ = 'molrepo_has_molecule'
    id = Column(Text, primary_key=True)
    molrepo_name = Column(Text, ForeignKey("molrepo.molrepo_name"), index=True)
    src_id = Column(Text)
    smiles = Column(Text)  # It means the source smiles
    inchikey = Column(VARCHAR(27), ForeignKey("molecule.inchikey"), index=True)

    molecule = relationship("Molecule", lazy='joined')

    def __repr__(self):
        """String representation."""
        return str(self.inchikey)

    @staticmethod
    def _create_table():
        engine = get_engine()
        Base.metadata.create_all(engine, tables=[MolrepoHasMolecule.__table__])

    @staticmethod
    def _drop_table():
        engine = get_engine()
        MolrepoHasMolecule.__table__.drop(engine)

    @staticmethod
    def _table_exists():
        engine = get_engine()
        return sqlalchemy.inspect(engine).has_table(
            MolrepoHasMolecule.__tablename__)

    @staticmethod
    def _table_attributes():
        attrs = [a for a in class_mapper(
            MolrepoHasMolecule).iterate_properties]
        col_attrs = [a.key for a in attrs if isinstance(a, ColumnProperty)]
        input_attrs = [a for a in col_attrs if a != 'id']
        return input_attrs

    @staticmethod
    def get(inchikey):
        """Get Molrepo entries associated to the given inchikey.

        Args:
            inchikey(str): The inchikey to search for.
        """
        session = get_session()
        query = session.query(MolrepoHasMolecule).filter_by(inchikey=inchikey)
        res = query.all()
        session.close()
        return res


@logged
class MolrepoHasDatasource(Base):
    """Molrepo-Datasource relationship.

    Many-to-Many relationship.
    """

    __tablename__ = 'molrepo_has_datasource'
    molrepo_name = Column(Text, ForeignKey("molrepo.molrepo_name"),
                          primary_key=True)
    datasource_name = Column(Text, ForeignKey("datasource.datasource_name"),
                             primary_key=True)

    def __repr__(self):
        """String representation."""
        return self.molrepo_name + " maps to " + self.datasource_name

    @staticmethod
    def _create_table():
        engine = get_engine()
        Base.metadata.create_all(
            engine, tables=[MolrepoHasDatasource.__table__])

    @staticmethod
    def _drop_table():
        engine = get_engine()
        MolrepoHasDatasource.__table__.drop(engine)

    @staticmethod
    def _table_exists():
        engine = get_engine()
        return sqlalchemy.inspect(engine).has_table(
            MolrepoHasDatasource.__tablename__)

    @staticmethod
    def _table_attributes():
        attrs = [a for a in class_mapper(
            MolrepoHasDatasource).iterate_properties]
        col_attrs = [a.key for a in attrs if isinstance(a, ColumnProperty)]
        input_attrs = [a for a in col_attrs if a != 'id']
        return input_attrs

    @staticmethod
    def add(kwargs):
        """Add a new row to the table.

        Args:
            kwargs(dict):The data in dictionary format.
        """
        if type(kwargs) is dict:
            entry = MolrepoHasDatasource(**kwargs)
        MolrepoHasDatasource.__log.debug(entry)
        session = get_session()
        session.add(entry)
        session.commit()
        session.close()

    @staticmethod
    def from_csv(filename):
        """Add entries from CSV file.

        Args:
            filename(str): Path to a CSV file.
        """
        import pandas as pd
        df = pd.read_csv(filename)
        # check columns
        needed_cols = MolrepoHasDatasource._table_attributes()
        if needed_cols != list(df.columns):
            raise Exception("Input missing columns: %s", ' '.join(needed_cols))
        # add them
        for row_nr, row in df.iterrows():
            try:
                MolrepoHasDatasource.add(row.dropna().to_dict())
            except Exception as err:
                MolrepoHasDatasource.__log.error(
                    "Error in line %s: %s", row_nr, str(err))
