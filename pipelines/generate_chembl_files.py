# Date: 25th June 2020
# Author Nicolas Soler
# Purpose: use the Chembl Python API to generate two TSV files required for the CC pipeline,
# namely chembl_drugtargets.txt and chembl_indications.txt

import os
import pandas as pd
from chembl_webresource_client.new_client import new_client

#-----------Generating chembl_drugtargets.txt
##----------fuctions

# Parsing the references of each record
def parseReference(refList):
    """Returns references separated by '####'"""
    out=[]
    for refDic in refList:
        # Each reference has fields separated by $$
        out.append(str(refDic['ref_type'])+'$$'+str(refDic['ref_id'])+'$$'+str(refDic['ref_url']))
    return '####'.join(out)

def yesNo(inputTxt):
    """Boolean fields in the txt are Y/N in the CSV"""
    if inputTxt == 'True' or inputTxt == True:
        return 'Y'
    elif inputTxt == 'False' or inputTxt == False:
        return 'N'
    else:
        return inputTxt


def checkNone(inputElem, clau, isList=False, subfield=None):

    if (not clau in inputElem) or (inputElem[clau] == None) or (inputElem[clau] == 'None'):
        return ''

    elif isList and not subfield:
        tmp= set([x.lower() for x in inputElem[clau]])
        return '; '.join(list(tmp))
    
    elif isList and subfield:
        tmp= list(set([dico[subfield].lower() for dico in inputElem[clau]]))
        return "; ".join(tmp)
    
    elif not isList and subfield:
        inputElem[clau][subfield]

    else:
        return inputElem[clau]

##----Main

def generate_chembl_files():

	outPutDir="/aloy/web_checker/repo_data"
	output=os.path.join(outPutDir, "chembldrugtargets.tsv")


	if not os.path.exists(output):
		### Extracting the relevant fields from drug
		outListDic=[]

		# Extracting the relevant fields from their drug database
		outListDic=[]

		drugs=new_client.drug
		print("Generating chembl_drugtargets.tsv")
		print("Number of records in chembl drug:",len(drugs))
		print("processing, please wait...")

		for elem in drugs:

		    outListDic.append({'CHEMBL_ID' : elem['molecule_chembl_id'],
		                    'SYNONYMS' : checkNone(elem, 'molecule_synonyms', subfield='molecule_synonym', isList=True),  #take the first one
		                   'DEVELOPMENT_PHASE' : checkNone(elem, 'development_phase'),
		                   'RESEARCH_CODES' : checkNone(elem, 'research_codes', isList=True),
		                   'APPLICANTS' : checkNone(elem, 'applicants', isList=True),
		                   'USAN_STEM' : checkNone(elem, 'usan_stem'),
		                   'USAN_STEM_DEFINITION': checkNone(elem, 'usan_stem_definition'),
		                   'USAN_STEM_SUBSTEM': checkNone(elem, 'usan_stem_substem'),           
		                   'USAN_YEAR': checkNone(elem, 'usan_year'),
		                   'FIRST_APPROVAL': checkNone(elem, 'first_approval'),                   
		                   'INDICATION_CLASS': checkNone(elem,'indication_class'),
		                   'SC_PATENT': checkNone(elem,'sc_patent'),
		                   'DRUG_TYPE': checkNone(elem,'drug_type'),
		                   'RULE_OF_FIVE': checkNone(elem,'rule_of_five',),
		                   'FIRST_IN_CLASS': checkNone(elem,'first_in_class'),
		                   'CHIRALITY': checkNone(elem,'chirality'),
		                   'PRODRUG': checkNone(elem,'prodrug'),
		                   'ORAL': checkNone(elem,'oral'),
		                   'PARENTERAL': checkNone(elem, 'parenteral'),
		                   'TOPICAL': checkNone(elem, 'topical'),
		                   'BLACK_BOX': checkNone(elem, 'black_box'),
		                   'TOPICAL': checkNone(elem, 'topical'),
		                   'AVAILABILITY_TYPE': checkNone(elem,'availability_type'),
		                   'WITHDRAWN_YEAR': checkNone(elem,'withdrawn_year'),
		                   'WITHDRAWN_COUNTRY': checkNone(elem,'withdrawn_country'),
		                   'WITHDRAWN_REASON': checkNone(elem,'withdrawn_reason'),
		                   'CANONICAL_SMILES': checkNone(elem,'molecule_structures',subfield='canonical_smiles'),                       
		                      })
		    
		    if 'atc_classification' in elem:
		        outListDic[-1]['ATC_CODE']= checkNone(elem,'atc_classification',subfield='code',isList=True)
		        outListDic[-1]['ATC_CODE_DESCRIPTION']= checkNone(elem,'atc_classification', subfield='description',isList=True)
		            
		    elif 'atc_code_description' in elem:
		        outListDic[-1]['ATC_CODE']= ''
		        outListDic[-1]['ATC_CODE_DESCRIPTION']= checkNone(elem,'atc_code_description')
		    
		    else:
		        outListDic[-1]['ATC_CODE']= ''
		        outListDic[-1]['ATC_CODE_DESCRIPTION']= ''
		    

		# From this point we can carry on with Pandas
		df=pd.DataFrame(outListDic)

		BooleanFields= ['RULE_OF_FIVE', 'FIRST_IN_CLASS', 'PRODRUG', 'ORAL', 'PARENTERAL', 'TOPICAL', 'BLACK_BOX']

		### Transforming True/False in Yes/No
		for f in BooleanFields:
		    df[f]=df[f].apply(yesNo)

		csv_header= ["CHEMBL_ID", "SYNONYMS", "DEVELOPMENT_PHASE", "RESEARCH_CODES", "APPLICANTS", "USAN_STEM", "USAN_STEM_DEFINITION", "USAN_STEM_SUBSTEM", "USAN_YEAR", "FIRST_APPROVAL", "ATC_CODE", "ATC_CODE_DESCRIPTION", "INDICATION_CLASS", "SC_PATENT", "DRUG_TYPE", "RULE_OF_FIVE", "FIRST_IN_CLASS", "CHIRALITY", "PRODRUG", "ORAL", "PARENTERAL", "TOPICAL", "BLACK_BOX", "AVAILABILITY_TYPE", "WITHDRAWN_YEAR", "WITHDRAWN_COUNTRY", "WITHDRAWN_REASON", "CANONICAL_SMILES"]

		### Saving the TSV The header of our output CSV file (We don't put PARENT_MOLREGNO)
		print("Writing", output)
		df[csv_header].to_csv(output,sep="\t")

	else:
		print("{} already exists, skipping".format(output))


	#-----------Generating chembl_drug_indication.txt

	output=os.path.join(outPutDir, "chembl_indications.tsv")

	if not os.path.exists(output):
		indications= new_client.drug_indication
		print("Generating chembldrugtargets.tsv")
		print("NUMBER OF RECORDS FOR INDICATION:",len(indications))
		print("processing, please wait...")

		## Main2
		outListDic=[]

		# Also Get a fixed list of CHEMBL_IDs to retrieve
		chembl_ids= set()

		for elem in indications:
		    outListDic.append({'MOLECULE_CHEMBL_ID' : elem['molecule_chembl_id'],
		                    'MESH_ID' : elem['mesh_id'],
		                   'MESH_HEADING' : elem['mesh_heading'],
		                   'EFO_ID' : elem['efo_id'],
		                   'EFO_NAME' : elem['efo_term'], 
		                   'MAX_PHASE_FOR_IND' : elem['max_phase_for_ind'],
		                   'REFS': parseReference(elem['indication_refs'])})
		    
		    chembl_ids.add(elem['molecule_chembl_id'])

		# From this point we can carry on with Pandas
		df=pd.DataFrame(outListDic)

		# Now we have to retrieve the missing fields from other databases
		# i.e: MOLECULE_NAME, MOLECULE_TYPE, FIRST_APPROVAL, USAN_YEAR
		chembl_ids=list(chembl_ids)
		print("Number of distinct Chembl_ids:",len(chembl_ids))

		# Filter the molecules we want
		fields_to_get= {'molecule_chembl_id': 'MOLECULE_CHEMBL_ID',
		                'pref_name':'MOLECULE_NAME',
		                'molecule_type':'MOLECULE_TYPE',
		                'first_approval': 'FIRST_APPROVAL',
		                'usan_year': 'USAN_YEAR'}

		molecules=new_client.molecule.filter(molecule_chembl_id__in=chembl_ids).only(list(fields_to_get.keys()))
		print("Number of distinct molecules:",len(molecules))

		# Adding the missing columns in the dataframe (can last several minutes)
		for f in list(fields_to_get.keys())[1:]:
		    new_field = fields_to_get[f]
		    df[new_field]=''
		    # Now change several entries at once in df
		    for mol in molecules:
		        df.loc[df.MOLECULE_CHEMBL_ID == mol['molecule_chembl_id'], new_field] = mol[f] 

		# Writing the TSV file
		print("writing", output)
		csv_header= ['MOLECULE_CHEMBL_ID', 'MOLECULE_NAME', 'MOLECULE_TYPE', "FIRST_APPROVAL", "MESH_ID", "MESH_HEADING", "EFO_ID", "EFO_NAME", "MAX_PHASE_FOR_IND", "USAN_YEAR", "REFS"]
		df[csv_header].to_csv( output,sep="\t")

	else:
		print("{} already exists, skipping".format(output))

##-----------------
if __name__=="__main__":
	generate_chembl_files()