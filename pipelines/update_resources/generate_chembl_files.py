# Date: 25th June 2020
# Author Nicolas Soler
# Purpose: use the Chembl Python API to generate two TSV files required for the CC pipeline,
# namely chembl_drugtargets.txt and chembl_indications.txt

import os
import signal
import time
import pandas as pd


# ------------------------ Functions ----------------------- #

# Parsing the references of each record
def parseReference(refList):
    """Returns references separated by '####'"""
    out = []
    for refDic in refList:
        # Each reference has fields separated by $$
        out.append(str(refDic['ref_type']) + '$$' +
                   str(refDic['ref_id']) + '$$' + str(refDic['ref_url']))
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
        tmp = set([x.lower() for x in inputElem[clau]])
        return '; '.join(list(tmp))

    elif isList and subfield:
        tmp = list(set([dico[subfield].lower() for dico in inputElem[clau]]))
        return "; ".join(tmp)

    elif not isList and subfield:
        return inputElem[clau][subfield]

    else:
        return inputElem[clau]


class _PageTimeout(Exception):
    pass


def _sigalrm_handler(signum, frame):
    raise _PageTimeout("page fetch timed out (SIGALRM)")


def _fetch_with_retry(get_queryset, label, page_timeout=20, max_retries=200):
    """Iterate a ChEMBL queryset with a hard per-page SIGALRM timeout and retry.

    get_queryset is a callable so a fresh queryset is created on each retry,
    avoiding corrupted internal pagination state after a failed iteration.
    Caching is left enabled so retried runs skip already-fetched pages.
    """
    signal.signal(signal.SIGALRM, _sigalrm_handler)

    for attempt in range(1, max_retries + 1):
        try:
            results = []
            it = iter(get_queryset())
            i = 0
            while True:
                signal.alarm(page_timeout)
                try:
                    elem = next(it)
                except StopIteration:
                    signal.alarm(0)
                    break
                signal.alarm(0)
                if i % 500 == 0:
                    print("  %s: %d records fetched..." % (label, i))
                results.append(elem)
                i += 1
            print("  %s: done, %d total records." % (label, len(results)))
            return results
        except (_PageTimeout, Exception) as e:
            signal.alarm(0)
            print("  %s: error on attempt %d/%d: %s" % (label, attempt, max_retries, e))
            if attempt == max_retries:
                raise
            print("  %s: retrying in 20s..." % label)
            time.sleep(20)
    return []

# -------------------------- Main ------------------------- #
# ----------- Generating chembl_drugtargets.txt ----------- #
def generate_chembl_files():
    from chembl_webresource_client.new_client import new_client

    outPutDir = "/aloy/web_checker/repo_data"
    output = os.path.join(outPutDir, "chembl_drugtargets.tsv")

    if not os.path.exists(output):
        print("Generating chembl_drugtargets.tsv")
        print("Number of records in chembl drug:", len(new_client.drug))

        drug_records = _fetch_with_retry(lambda: new_client.drug, "drugs")

        outListDic = []
        for elem in drug_records:
            outListDic.append({'CHEMBL_ID': elem['molecule_chembl_id'],
                               # take the first one
                               'SYNONYMS': checkNone(elem, 'molecule_synonyms', subfield='molecule_synonym', isList=True),
                               'DEVELOPMENT_PHASE': checkNone(elem, 'development_phase'),
                               'RESEARCH_CODES': checkNone(elem, 'research_codes', isList=True),
                               'APPLICANTS': checkNone(elem, 'applicants', isList=True),
                               'USAN_STEM': checkNone(elem, 'usan_stem'),
                               'USAN_STEM_DEFINITION': checkNone(elem, 'usan_stem_definition'),
                               'USAN_STEM_SUBSTEM': checkNone(elem, 'usan_stem_substem'),
                               'USAN_YEAR': checkNone(elem, 'usan_year'),
                               'FIRST_APPROVAL': checkNone(elem, 'first_approval'),
                               'INDICATION_CLASS': checkNone(elem, 'indication_class'),
                               'SC_PATENT': checkNone(elem, 'sc_patent'),
                               'DRUG_TYPE': checkNone(elem, 'drug_type'),
                               'RULE_OF_FIVE': checkNone(elem, 'rule_of_five',),
                               'FIRST_IN_CLASS': checkNone(elem, 'first_in_class'),
                               'CHIRALITY': checkNone(elem, 'chirality'),
                               'PRODRUG': checkNone(elem, 'prodrug'),
                               'ORAL': checkNone(elem, 'oral'),
                               'PARENTERAL': checkNone(elem, 'parenteral'),
                               'TOPICAL': checkNone(elem, 'topical'),
                               'BLACK_BOX': checkNone(elem, 'black_box'),
                               'TOPICAL': checkNone(elem, 'topical'),
                               'AVAILABILITY_TYPE': checkNone(elem, 'availability_type'),
                               'WITHDRAWN_YEAR': checkNone(elem, 'withdrawn_year'),
                               'WITHDRAWN_COUNTRY': checkNone(elem, 'withdrawn_country'),
                               'WITHDRAWN_REASON': checkNone(elem, 'withdrawn_reason'),
                               'CANONICAL_SMILES': checkNone(elem, 'molecule_structures', subfield='canonical_smiles'),
                               })

            if 'atc_classification' in elem:
                outListDic[-1]['ATC_CODE'] = checkNone(
                    elem, 'atc_classification', subfield='code', isList=True)
                outListDic[-1]['ATC_CODE_DESCRIPTION'] = checkNone(
                    elem, 'atc_classification', subfield='description', isList=True)

            elif 'atc_code_description' in elem:
                outListDic[-1]['ATC_CODE'] = ''
                outListDic[-1]['ATC_CODE_DESCRIPTION'] = checkNone(
                    elem, 'atc_code_description')

            else:
                outListDic[-1]['ATC_CODE'] = ''
                outListDic[-1]['ATC_CODE_DESCRIPTION'] = ''

        # From this point we can carry on with Pandas
        df = pd.DataFrame(outListDic)

        BooleanFields = ['RULE_OF_FIVE', 'FIRST_IN_CLASS',
                         'PRODRUG', 'ORAL', 'PARENTERAL', 'TOPICAL', 'BLACK_BOX']

        # Transforming True/False in Yes/No
        for f in BooleanFields:
            df[f] = df[f].apply(yesNo)

        csv_header = ["CHEMBL_ID", "SYNONYMS", "DEVELOPMENT_PHASE", "RESEARCH_CODES", "APPLICANTS", "USAN_STEM", "USAN_STEM_DEFINITION", "USAN_STEM_SUBSTEM", "USAN_YEAR", "FIRST_APPROVAL", "ATC_CODE", "ATC_CODE_DESCRIPTION", "INDICATION_CLASS",
                      "SC_PATENT", "DRUG_TYPE", "RULE_OF_FIVE", "FIRST_IN_CLASS", "CHIRALITY", "PRODRUG", "ORAL", "PARENTERAL", "TOPICAL", "BLACK_BOX", "AVAILABILITY_TYPE", "WITHDRAWN_YEAR", "WITHDRAWN_COUNTRY", "WITHDRAWN_REASON", "CANONICAL_SMILES"]

        # Saving the TSV The header of our output CSV file (We don't put
        # PARENT_MOLREGNO)
        print("Writing", output)
        df[csv_header].to_csv(output, sep="\t")

    else:
        print("{} already exists, skipping".format(output))

    #-----------Generating chembl_drug_indication.txt

    output = os.path.join(outPutDir, "chembl_indications.tsv")

    if not os.path.exists(output):
        print("Generating chembl_indications.tsv")
        print("NUMBER OF RECORDS FOR INDICATION:", len(new_client.drug_indication))

        indication_records = _fetch_with_retry(lambda: new_client.drug_indication, "indications")

        # Main2
        outListDic = []

        # Also Get a fixed list of CHEMBL_IDs to retrieve
        chembl_ids = set()

        for elem in indication_records:
            outListDic.append({'MOLECULE_CHEMBL_ID': elem['molecule_chembl_id'],
                               'MESH_ID': elem['mesh_id'],
                               'MESH_HEADING': elem['mesh_heading'],
                               'EFO_ID': elem['efo_id'],
                               'EFO_NAME': elem['efo_term'],
                               'MAX_PHASE_FOR_IND': elem['max_phase_for_ind'],
                               'REFS': parseReference(elem['indication_refs'])})

            chembl_ids.add(elem['molecule_chembl_id'])

        # From this point we can carry on with Pandas
        df = pd.DataFrame(outListDic)

        # Now we have to retrieve the missing fields from other databases
        # i.e: MOLECULE_NAME, MOLECULE_TYPE, FIRST_APPROVAL, USAN_YEAR
        chembl_ids = list(chembl_ids)
        print("Number of distinct Chembl_ids:", len(chembl_ids))

        # Filter the molecules we want
        fields_to_get = {'molecule_chembl_id': 'MOLECULE_CHEMBL_ID',
                         'pref_name': 'MOLECULE_NAME',
                         'molecule_type': 'MOLECULE_TYPE',
                         'first_approval': 'FIRST_APPROVAL',
                         'usan_year': 'USAN_YEAR'}

        api_fields = list(fields_to_get.keys())
        molecule_records = _fetch_with_retry(
            lambda: new_client.molecule.filter(
                molecule_chembl_id__in=chembl_ids).only(api_fields),
            "molecules")
        print("Number of distinct molecules:", len(molecule_records))

        # Adding the missing columns in the dataframe in a single pass
        mol_lookup = {mol['molecule_chembl_id']: mol for mol in molecule_records}
        for f in api_fields[1:]:
            new_field = fields_to_get[f]
            df[new_field] = df['MOLECULE_CHEMBL_ID'].map(
                lambda cid: mol_lookup.get(cid, {}).get(f, ''))

        # Writing the TSV file
        print("writing", output)
        csv_header = ['MOLECULE_CHEMBL_ID', 'MOLECULE_NAME', 'MOLECULE_TYPE', "FIRST_APPROVAL",
                      "MESH_ID", "MESH_HEADING", "EFO_ID", "EFO_NAME", "MAX_PHASE_FOR_IND", "USAN_YEAR", "REFS"]
        df[csv_header].to_csv(output, sep="\t")

    else:
        print("{} already exists, skipping".format(output))


if __name__ == "__main__":
    generate_chembl_files()
