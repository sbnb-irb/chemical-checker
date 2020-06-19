from chemicalchecker.database import Datasource

CSVfileIn="configs/datasource_19Jun2020.csv"

# check if Datasource table is there
if not Datasource._table_exists():
    # create the Datasource table
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
