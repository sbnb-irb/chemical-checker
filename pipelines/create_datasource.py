# Nico, 19/06/2020
# Update the datasource table from cc_package database on aloy-dbsrv
# Don't forget to remove the previous datasource table first

from chemicalchecker.database import Datasource
import os

CSVfileIn="configs/current_resource_CSV"
sql_file="configs/foreign_constraint.sql"

current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['CC_CONFIG'] = os.path.join(current_dir,'configs/cc_package.json')

# check if Datasource table is there
if Datasource._table_exists():
    print("Removing previous table 'datasource' in database 'cc_package'")
    Datasource._drop_table()

# create the Datasource table
print("Creating the table 'datasource' in database 'cc_package'")
Datasource._create_table()

# populate it with Datasources needed for exemplary Datasets
Datasource.from_csv(CSVfileIn)
print("TABLE CREATED---->datasource")

# Ensure the foreign key constraints remain as such (no password needed for sbnb-adm)
command="psql -h aloy-dbsrv -d cc_package <"+sql_file
os.system(command)