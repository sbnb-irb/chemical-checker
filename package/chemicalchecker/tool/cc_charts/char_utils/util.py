
def query_to_inchikey(query):
    """Detects the type of query and converts it to an inchikey."""

    from chemicalchecker.util.keytype import KeyTypeDetector
    from chemicalchecker.util.parser import Converter

    kd = KeyTypeDetector('')
    keytype = kd.type(query)

    if keytype is None:
        smi = Converter().chemical_name_to_smiles(query)
        inchikey = Converter().smiles_to_inchi(smi)[0]

    elif keytype=='inchikey':
        inchikey = query

    elif keytype=='smiles':
        inchikey = Converter().smiles_to_inchi(query)[0]   
    print(inchikey)
    return keytype, inchikey
