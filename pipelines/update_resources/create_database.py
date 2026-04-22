from chemicalchecker.database import Datasource, Dataset, DatasetHasDatasource, \
                                Molrepo, MolrepoHasDatasource, MolrepoHasMolecule, Molecule
from chemicalchecker.util import Config
import os

#DatasourceFile="datasource.csv"
DatasourceFile="current_resources.csv"
DatasetFile="dataset.csv"
DatasetDatasourceFile="dataset_has_datasource.csv"
MolrepoFile="molrepo.csv"
MolrepoDatasourceFile="molrepo_has_datasource.csv"

current_dir = os.path.dirname(os.path.abspath(__file__))

def create_db_dataset():
    flag = True
    
    # Create DB
    command1 ="PGPASSWORD={} psql -h {} -U {} -tc \"SELECT 1 FROM pg_database WHERE datname = \'{}\';\" | grep -q 1" \
                .format(Config().DB.password, Config().DB.host, Config().DB.user, Config().DB.database)
    res = os.system(command1)
    if res == 0:
        flag=False
        #raise Exception("Database '{}' already exists".format(Config().DB.database))
    else:
        command1 = "PGPASSWORD={} psql -h {} -U {} -c \"CREATE DATABASE {}\"".format(Config().DB.password, Config().DB.host, Config().DB.user, Config().DB.database)
        os.system(command1)
        print("DB CREATED---->{}".format(Config().DB.database))
    
    if( flag ):
        # check if datasource table is already present
        if Datasource._table_exists():
            Datasource._drop_table()
            print("'datasource' table already exists in database '{}' \n Dropped it".format(Config().DB.database))    
        print("Creating table 'datasource' in database '{}'".format(Config().DB.database))
        Datasource._create_table()
        print("Populating 'datasource' table with data")
        Datasource.from_csv(os.path.join(current_dir, DatasourceFile))
        print("TABLE CREATED---->datasource")

        # check if dataset table is already present
        if Dataset._table_exists():
            print("'dataset' table already exists in database '{}' \n Dropped it ".format(Config().DB.database))    
        print("Creating table 'dataset' in database '{}'".format(Config().DB.database))
        Dataset._create_table()
        print("Populating 'dataset' table with data")
        Dataset.from_csv(os.path.join(current_dir,DatasetFile))
        print("TABLE CREATED---->dataset")

        # check if dataset_has_datasource table is already present
        if DatasetHasDatasource._table_exists():
            DatasetHasDatasource._drop_table()
            print("'dataset_has_datasource' table already exists in database '{}' \n Dropped it".format(Config().DB.database))
        dataset_exists = Dataset._table_exists()
        datasource_exists = Datasource._table_exists()
        if dataset_exists and datasource_exists:    
            print("Creating table 'dataset_has_datasource' in database '{}'".format(Config().DB.database))
            DatasetHasDatasource._create_table()
            print("Populating 'dataset_has_datasource' table with data")
            DatasetHasDatasource.from_csv(os.path.join(current_dir,DatasetDatasourceFile))
            print("TABLE CREATED---->dataset_has_datasource")
        else:
            raise Exception("It is not possble to create 'dataset_has_datasource' because either 'dataset' or 'datasource' table doesn't exist: \
                dataset {} - datasource {}".format(dataset_exists, datasource_exists)) 

        # check if molrepo table is already present
        if Molrepo._table_exists():
            Molrepo._drop_table()
            print("'molrepo' table already exists in database '{}' \n Dropped it".format(Config().DB.database))
        print("Creating table 'molrepo' in database '{}'".format(Config().DB.database))
        Molrepo._create_table()
        print("Populating 'molrepo' table with data")
        Molrepo.from_csv(os.path.join(current_dir,MolrepoFile))
        print("TABLE CREATED---->molrepo")

        # check if molrepo_has_datasource table is already present
        if MolrepoHasDatasource._table_exists():
            MolrepoHasDatasource._drop_table()
            print("'molrepo_has_datasource' table already exists in database '{}' \n Dropped it".format(Config().DB.database))
        molrepo_exists = Molrepo._table_exists()
        datasource_exists = Datasource._table_exists()
        if molrepo_exists and datasource_exists:        
            print("Creating table 'molrepo_has_datasource' in database '{}'".format(Config().DB.database))
            MolrepoHasDatasource._create_table()
            print("Populating 'molrepo_has_datasource' table with data")
            MolrepoHasDatasource.from_csv(os.path.join(current_dir,MolrepoDatasourceFile))
            print("TABLE CREATED---->molrepo_has_datasource")
        else:
            raise Exception("Itis not possble to create 'molrepo_has_datasource' because either 'molrepo' or 'datasource' table doesn't exist: \
                molrepo {} - datasource {}".format(molrepo_exists, datasource_exists)) 

        # molecule and molrepo_has_molecule tables are created and left empty until pipeline step 'molrepo'
        if Molecule._table_exists():
            Molecule._drop_table()
            print("'molecule' table already exists in database '{}' \n Dropped it".format(Config().DB.database))
        print("Creating table 'molecule' in database '{}'".format(Config().DB.database))
        Molecule._create_table()

        if MolrepoHasMolecule._table_exists():
            MolrepoHasMolecule._drop_table()
            print("'molrepo_has_molecule' table already exists in database '{}' \n Dropped it".format(Config().DB.database))
        molrepo_exists = Molrepo._table_exists()
        molecule_exists = Molecule._table_exists()
        if molrepo_exists and molecule_exists: 
            print("Creating table 'molrepo_has_molecule' in database '{}'".format(Config().DB.database))
            MolrepoHasMolecule._create_table()
        else:
            raise Exception("Itis not possble to create 'molrepo_has_molecule' because either 'molrepo' or 'molecule' table doesn't exist: \
                molrepo {} - molecule {}".format(molrepo_exists, molecule_exists)) 

if __name__ == '__main__':
    create_db_dataset()
