# Nico, 19/06/2020
# Update the datasource table from cc_package database on aloy-dbsrv
# Don't forget to remove the prvious datasource table first

from chemicalchecker.database import Datasource
import os

CSVfileIn="configs/current_resource_CSV"

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



# start 45 download jobs (one per Datasource), job will wait until finished
#job = Datasource.download_hpc('/aloy/scratch/sbnb-adm/tmp_job_download')
# check if the downloads are really done
#if not Datasource.test_all_downloaded():
#    print("Something went WRONG while DOWNLOAD, should retry")
    # print the faulty one
#    for ds in Datasource.get():
#        if not ds.available():
#            print("Datasource %s not available" % ds)
