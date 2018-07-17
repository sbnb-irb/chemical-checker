#
# General configuration for the Chemical Checker pipeline
#

# Imports
import os
import sys
import getpass
if sys.version_info[0] >= 3:
  from configparser import RawConfigParser
else:
  from ConfigParser import RawConfigParser

# Constants

# Main log filename
logFilename = 'pipeline.log'


# Chemical Checker sender
checkerEmailAddress = ("Chemical Checker","checker@irbbarcelona.org")

# List of steps
steps = [
  "0_downloads",
  "1_raws"
]

mosaic = "mosaic"
chembl = "chembl"

dbname = "cchecker"

drugbank_download = "full database.xml"
pdb_components_smiles_download = "Components-smiles-stereo-oe.smi"
bindingdb_download =  "BindingDB_All.tsv"
chebi_lite_download = "ChEBI_lite_3star.sdf"
smpdb_folder_download =  "smpdb_structures"
lincs_GSE92742_pert_info_download =  "GSE70138_Broad_LINCS_pert_info_*.txt"
lincs_GSE70138_pert_info_download =  "GSE92742_Broad_LINCS_pert_info.txt"
nci60_download = "DTP_NCI60_ZSCORE.csv"
mosaic_all_collections_download = "All_Collections.sdf"
morphlincs_molecules_download = "LDS-1195/Metadata/Small_Molecule_Metadata.txt"
kegg_atcs_download =  "br08303.keg"
kegg_mol_folder_download  = "kegg/"
stitch_molecules_download =  "chemicals.v4.0.tsv"
sider_download = "meddra_all_se.tsv"
ctd_molecules_download = "ctd.smi"
eco_domains = "ecod.latest.domains.txt"
chebi_obo = 'chebi.obo'
pathway_data        = 'PathwayCommons9.All.hgnc.txt.gz'
hgnc_mapping_file = 'hgnc_mapping.txt'
string_tab_file = 'HUMAN_9606_idmapping_selected.tab'
string_network_file = 'protein.links.txt'
pathway_sif = "all_binary.sif"
id_conversion = "id_conversion.txt"
file_9606 = "9606.txt"
human_proteome = 'human_proteome.tab'
uniprot2reactome = 'UniProt2Reactome_All_Levels.txt'
nci60_zcore = 'DTP_NCI60_ZSCORE.csv'
all_conditions = 'All_conditions.txt'
comb_gt_preds = 'combined_gene-target-predictions.txt'
cellosaurus_obo = 'cellosaurus.obo'
repodb = 'repodb.csv'
ctd_diseases = 'CTD_diseases.tsv'
umls_disease_mappings = 'disease_mappings.tsv'
sider_file = 'meddra_all_se.tsv'
chemdis_file = 'CTD_chemicals_diseases.tsv'
go_file = 'go-basic.obo'
goa_human= 'goa_human.gaf'


#Downloads (link,username, password,outputfile)

downloads = [('ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_*_postgresql.tar.gz','','','chembl_postgresql.tar.gz'),
             ('https://www.ebi.ac.uk/chembl/download_helper/drugstore_txt','','','chembl_drugtargets.txt'),
             ('https://www.ebi.ac.uk/chembl/download_helper/indications_txt','','','chembl_indications.txt'),
             ('https://www.drugbank.ca/releases/5-1-0/downloads/all-full-database','oriol.guitart@irbbarcelona.org','sbnbAloy','drugbank_all_full_database.xml.zip'),
             ('https://www.bindingdb.org/bind/downloads/BindingDB_All_2018m3.tsv.zip','','','BindingDB_All.tsv.zip'),
             ('http://prodata.swmed.edu/ecod/distributions/ecod.latest.domains.txt','','',eco_domains),
             ('http://ligand-expo.rcsb.org/dictionaries/cc-to-pdb.tdd','','','cc-to-pdb.tdd'),
             ('http://ligand-expo.rcsb.org/dictionaries/Components-smiles-stereo-oe.smi','','','Components-smiles-stereo-oe.smi'),
             ('ftp://ftp.ebi.ac.uk/pub/databases/chebi/SDF/ChEBI_lite_3star.sdf.gz','','','ChEBI_lite_3star.sdf.gz'),
             ('ftp://ftp.ebi.ac.uk/pub/databases/chebi/Flat_file_tab_delimited/compounds_3star.tsv.gz','','','compounds_3star.tsv.gz'),
             ('ftp://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.obo','','',chebi_obo),
             ('ftp://phylomedb.org/metaphors/latest/id_conversion.txt.gz','','',id_conversion + '.gz'),
             ('ftp://phylomedb.org/metaphors/latest/orthologs/9606.txt.gz','','',file_9606 + '.gz'),
             ('https://reactome.org/download/current/ChEBI2Reactome_All_Levels.txt','','','ChEBI2Reactome_All_Levels.txt'),
             ('https://reactome.org/download/current/UniProt2Reactome_All_Levels.txt','','',uniprot2reactome),
             ('https://reactome.org/download/current/ReactomePathwaysRelation.txt','','','ReactomePathwaysRelation.txt'),
             ('https://www.uniprot.org/uniprot/?query=proteome:UP000005640&format=tab','','',human_proteome),
             ('ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz','','',goa_human +'.gz'),
             ('http://snapshot.geneontology.org/ontology/go-basic.obo','','',go_file),
             ('http://smpdb.ca/downloads/smpdb_structures.zip','','','smpdb_structures.zip'),
             ('ftp://ftp.ebi.ac.uk/pub/databases/genenames/new/tsv/hgnc_complete_set.txt','','',hgnc_mapping_file),
             ('ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping_selected.tab.gz','','',string_tab_file+'.gz'),
             ('http://www.pathwaycommons.org/archives/PC2/v9/PathwayCommons9.All.hgnc.txt.gz','','',pathway_data),
             ('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_sig_info*.txt.gz','','','GSE70138_Broad_LINCS_sig_info*.txt.gz'),
             ('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_info.txt.gz','','','GSE92742_Broad_LINCS_sig_info.txt.gz'),
             ('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_gene_info*.txt.gz','','','GSE70138_Broad_LINCS_gene_info*.txt.gz'),
             ('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_sig_metrics*.txt.gz','','','GSE70138_Broad_LINCS_sig_metrics*.txt.gz'),
             ('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_inst_info*.txt.gz','','','GSE70138_Broad_LINCS_inst_info*.txt.gz'),
             ('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_metrics.txt.gz','','','GSE92742_Broad_LINCS_sig_metrics.txt.gz'),
             ('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_gene_info.txt.gz','','','GSE92742_Broad_LINCS_gene_info.txt.gz'),
             ('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328*.gctx.gz','','','GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328*.gctx.gz'),
             ('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_pert_info_*.txt.gz','','','GSE70138_Broad_LINCS_pert_info_*.txt.gz'),
             ('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_pert_info.txt.gz','','','GSE70138_Broad_LINCS_pert_info.txt.gz'),
             ('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz','','','GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz'),
             ('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_pert_info.txt.gz','','','GSE92742_Broad_LINCS_pert_info.txt.gz'),
             ('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_pert_metrics.txt.gz','','','GSE92742_Broad_LINCS_pert_metrics.txt.gz'),
             ('https://discover.nci.nih.gov/cellminerdata/normalizedArchives/DTP_NCI60_ZSCORE.zip','','','DTP_NCI60_ZSCORE.zip'),
             ('http://mosaic.cs.umn.edu/downloads/RIKEN-Clinical_FINAL/tables/combined/combined_gene-target-predictions.zip','','','combined_gene-target-predictions.zip'),
             ('http://mosaic.cs.umn.edu/downloads/RIKEN-Clinical_FINAL/matrices_and_cdts/matrices/All_conditions.txt','','',all_conditions),
             ('http://mosaic.cs.umn.edu/downloads/RIKEN-Clinical_FINAL/compound_information/sdf/Structural_Data_Files_combined.zip','','','Structural_Data_Files_combined.zip'),
             ('http://lincsportal.ccs.miami.edu/dcic/api/download?path=LINCS_Data/Broad_Therapeutics&file=LDS-1195.tar.gz','','','LDS-1195.tar.gz'),
             ('ftp://ftp.expasy.org/databases/cellosaurus/cellosaurus.obo','','',cellosaurus_obo),
             ('http://www.genome.jp/kegg-bin/download_htext?htext=br08303.keg&format=htext&filedir=','','',kegg_atcs_download),
             ('https://chiragjp.shinyapps.io/repoDB/_w_bb51f2e4/session/4ea0b89b04d865cf86f6ba1bba3feafe/download/downloadFull?w=bb51f2e4','','',repodb),
             ('http://ctdbase.org/reports/CTD_chemicals.tsv.gz','','','CTD_chemicals.tsv.gz'),
             ('http://ctdbase.org/reports/CTD_diseases.tsv.gz','','',ctd_diseases + '.gz'),
             ('http://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz','','','CTD_chemicals_diseases.tsv.gz'),
             ('http://www.disgenet.org/ds/DisGeNET/results/disease_mappings.tsv.gz','','',umls_disease_mappings + '.gz'),
             ('http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz','','',sider_file + '.gz'),
             ('http://sideeffects.embl.de/media/download/meddra_freq.tsv.gz','','','meddra_freq.tsv.gz'),
             ('http://stitch4.embl.de/download/chemicals.v4.0.tsv.gz','','','chemicals.v4.0.tsv.gz')
             ]

# Directories inside the output directory
DATASETS_SUBDIR           = "datasets"


DOWNLOAD_SUBDIR           = "downloads"
WEBREPO        = '/aloy/web_checker/'
WEBREPOMOLS    = WEBREPO + "/molecules/"
MOSAICPATH     = "/aloy/home/mduran/myscripts/mosaic/"
MOLREPO        = "molrepo"
NETWORKS        = "networks"

LOG_SUBDIR                = "log"
READY_SUBDIR              = "ready"
TMP_SUBDIR                = "tmp"


KEEP_TABLES = ['structure','fp2d','fp3d','scaffolds','subskeys','physchem','pubchem']

# Other directories
LOCAL_DIR                 = "/local/"+getpass.getuser()

# Stuff related to the pac cluster and the submission of jobs
MASTERNODE          = "pac-one-head.irb.pcb.ub.es"

currentDir = os.path.dirname(os.path.abspath( __file__ ))

HOTNET_PATH = os.path.join(currentDir,"../../tools/hierarchical-hotnet/src/")

SETUPARRAYJOBMOD    = os.path.join(currentDir,"../../src/utils/setupArrayJobMod.py -q -x -N %(JOB_NAME)s -t %(NUM_TASKS)d -l %(TASKS_LIST)s %(COMMAND)s")
SETUPARRAYJOBNOLIST = os.path.join(currentDir,"../../src/utils/setupArrayJobNoJobList.py -q -x -N %(JOB_NAME)s -t %(NUM_TASKS)d %(COMMAND)s")
SETUPARRAYJOB       = os.path.join(currentDir,"../../src/utils/setupArrayJob.py -q -x -N %(JOB_NAME)s -t %(NUM_TASKS)d -l %(TASKS_LIST)s %(COMMAND)s")

SUBMITJOB           = "qsub -sync y "

SING_IMAGE = '/aloy/home/sbnb-adm/singularity-images/ubuntu-checker.simg'
# Intervals used when polling for results
# One minute = 60 seconds
POLL_TIME_INTERVAL = 60
WAIT_TIME_INTERVAL = 30
# Ten minutes = 600 seconds
LONG_POLL_TIME_INTERVAL = 60



# Classes
class checkerConf:
  """Reads and handles the configuration for Chemical Checker. Returns the name
     of the standard output directories and commands."""
  
  _configParser = None
  _OUTPUTDIR    = None
  _DBSDIR       = None
  _RUN          = None
  _SCRATCHDIR   = None
  _MODBASE      = None
  _PDBFILES     = None
  
  def __init__( self , configFilename):
    self._configParser = RawConfigParser()
    self._configParser.read(configFilename)
    self._VERSION_NUMBER  = self._configParser.get('General', 'release')
    self._SCRATCHDIR =  self._configParser.get('General', 'scratchdir')
 
  
  def hasVariable( self, section, variable ):
    return self._configParser.has_option( section, variable )
  
  def getVariable( self, section, variable ):
    return self._configParser.get( section, variable )


  def getDirectory( self, dirSpec ):
    if dirSpec == "downloads":
      #os.path.join(self._OUTPUTDIR,DATASETS_SUBDIR,dataset,INTERACTOME_SUBDIR)
      return os.path.join(self._SCRATCHDIR,self._VERSION_NUMBER,DOWNLOAD_SUBDIR)
  
    if dirSpec == "scratch":  
      return os.path.join(self._SCRATCHDIR,self._VERSION_NUMBER)

    if dirSpec == "logs":  
      return os.path.join(self._SCRATCHDIR,self._VERSION_NUMBER,LOG_SUBDIR)
  
    if dirSpec == "ready":  
      return os.path.join(self._SCRATCHDIR,self._VERSION_NUMBER,READY_SUBDIR)
  
    if dirSpec == "molRepo":  
      return os.path.join(self._SCRATCHDIR,self._VERSION_NUMBER,MOLREPO)
  
    if dirSpec == "temp":  
      return os.path.join(self._SCRATCHDIR,self._VERSION_NUMBER,TMP_SUBDIR)
  
    if dirSpec == "networks":  
      return os.path.join(self._SCRATCHDIR,self._VERSION_NUMBER,NETWORKS)
    

    raise Exception("Request for unknown directory %s" % dirSpec )
