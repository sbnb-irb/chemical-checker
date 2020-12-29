"""Database interaction via ORM (Object-relational-mapping).

The goal of this module is to interact with the local database system
and provides methods to access and query each of the database tables,
without the use of any kind of hard-coded query.

The package is database agnostic, that is, the database could be any kind
of database technologies and the user/programmer does not need to be aware of
it.

For this purpose we adopt a `ORM (Object-relational-mapping)` approach, that
is, each table record are represente as instances of python objects.

For example::

    from chemicalchecker.database import Dataset
    ds = Dataset.get('A1.001')
    ds.description

    >>> 'Binary representation of the 2D structure of a molecule....'

So, the `ds` instance of :class:`Dataset` exposes attributes that are the
fields of the datbase table.
"""
from .database import get_engine, get_session, set_db_config
from .dataset import Dataset, DatasetHasDatasource
from .pubchem import Pubchem
from .molecule import Molecule
from .datasource import Datasource
from .molrepo import Molrepo, MolrepoHasDatasource, MolrepoHasMolecule
from .calcdata import Calcdata
from .uniprotkb import UniprotKB, UniprotKbError
