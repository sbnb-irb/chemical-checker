"""Fetch PharmacoDB data from GraphQL API and populate local PostgreSQL tables.

Queries https://pharmacodb.ca/graphql and populates 3 tables in the
'pharmacodb' database on the configured PostgreSQL server:

    drug_annots  (drug_id, smiles, pubchem)
    genes        (gene_id, gene_name)
    gene_drugs   (drug_id, gene_id, estimate, pvalue, mDataType)

This recreates the tables expected by:
  - package/chemicalchecker/util/parser/parser.py :: pharmacodb()
  - package/scripts/preprocess/D2.002/run.py
"""
import os
import logging
import argparse
import requests
import requests.packages.urllib3
import psycopg2
from concurrent.futures import ThreadPoolExecutor, as_completed

requests.packages.urllib3.disable_warnings(
    requests.packages.urllib3.exceptions.InsecureRequestWarning
)

log = logging.getLogger(__name__)

GRAPHQL_URL = "https://pharmacodb.ca/graphql"
DB_NAME = "pharmacodb"
MAX_WORKERS = 20
BATCH_SIZE = 500

# mDataType values that represent mRNA expression in the new API
MRNA_TYPES = {"Kallisto_0.46.1.rnaseq", "rna"}


def _gql(query, retries=3):
    """Execute a GraphQL query against the PharmacoDB API.

    Args:
        query (str): GraphQL query string.
        retries (int): Number of attempts before raising on failure.

    Returns:
        dict: The ``data`` field of the GraphQL JSON response.

    Raises:
        RuntimeError: If the response contains GraphQL-level errors.
        requests.HTTPError: If the HTTP request fails after all retries.
    """
    for attempt in range(retries):
        try:
            # verify=False: conda's OpenSSL may fail TLS handshake with some hosts
            # while system curl succeeds; safe here as we are only reading public data
            resp = requests.post(
                GRAPHQL_URL,
                json={"query": query},
                timeout=60,
                verify=False
            )
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                raise RuntimeError("GraphQL error: %s" % data["errors"])
            return data["data"]
        except Exception as e:
            if attempt == retries - 1:
                raise
            log.warning("GraphQL attempt %d failed: %s — retrying", attempt + 1, e)


def _get_conn_params(config):
    """Extract psycopg2-compatible connection parameters from a CC Config object.

    Strips keys that are not accepted by psycopg2 (``dialect``,
    ``calcdata_dbname``, ``uniprot_db_version``).

    Args:
        config: A ``chemicalchecker.util.Config`` instance.

    Returns:
        dict: Connection parameters ready to pass to ``psycopg2.connect``.
    """
    params = config.DB.asdict()
    params.pop("dialect", None)
    params.pop("calcdata_dbname", None)
    params.pop("uniprot_db_version", None)
    return params


def _ensure_db(config):
    """Create the ``pharmacodb`` PostgreSQL database if it does not exist.

    Connects to the default database specified in the config and issues
    ``CREATE DATABASE`` with autocommit (isolation level 0) so the statement
    runs outside a transaction block, as PostgreSQL requires.

    Args:
        config: A ``chemicalchecker.util.Config`` instance.
    """
    params = _get_conn_params(config)
    conn = psycopg2.connect(**params)
    conn.set_isolation_level(0)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
    if not cur.fetchone():
        cur.execute("CREATE DATABASE %s" % DB_NAME)
        log.info("Created database '%s'", DB_NAME)
    else:
        log.info("Database '%s' already exists", DB_NAME)
    conn.close()


def _connect(config):
    """Open a psycopg2 connection to the ``pharmacodb`` database.

    Args:
        config: A ``chemicalchecker.util.Config`` instance.

    Returns:
        psycopg2 connection object.
    """
    params = _get_conn_params(config)
    params["database"] = DB_NAME
    return psycopg2.connect(**params)


def _create_tables(conn):
    """Drop and recreate the three PharmacoDB tables.

    Tables created:
        - ``drug_annots (drug_id PK, smiles, pubchem)`` — queried by
          ``parser.py::pharmacodb()`` to register molecules.
        - ``genes (gene_id PK, gene_name)`` — queried by D2.002 to resolve
          gene symbols used as feature labels.
        - ``gene_drugs (drug_id FK, gene_id FK, estimate, pvalue, mDataType)``
          — queried by D2.002 for drug-gene correlation signatures.

    Dropping in dependency order (gene_drugs first) before recreating ensures
    the operation is idempotent across re-runs.

    Args:
        conn: An open psycopg2 connection to ``pharmacodb``.
    """
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS gene_drugs")
    cur.execute("DROP TABLE IF EXISTS drug_annots")
    cur.execute("DROP TABLE IF EXISTS genes")
    cur.execute("""
        CREATE TABLE drug_annots (
            drug_id  INTEGER PRIMARY KEY,
            smiles   TEXT,
            pubchem  TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE genes (
            gene_id   INTEGER PRIMARY KEY,
            gene_name TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE gene_drugs (
            drug_id     INTEGER REFERENCES drug_annots(drug_id),
            gene_id     INTEGER REFERENCES genes(gene_id),
            estimate    FLOAT,
            pvalue      FLOAT,
            "mDataType" TEXT
        )
    """)
    conn.commit()
    log.info("Tables created")


def _create_indexes(conn):
    """Build indexes on ``gene_drugs`` after bulk data load.

    Indexes are created after all inserts so that Postgres builds each index in
    a single pass rather than maintaining it row-by-row during the load.

    Indexes created on ``gene_drugs``:
        - ``(mDataType, pvalue, gene_id, drug_id, estimate)`` — covering index
          for the D2.002 query: equality filter first, range filter second,
          remaining selected columns after. Includes every column the query
          touches so Postgres can do an index-only scan without heap fetches.
        - ``(drug_id)`` — separate index for FK constraint checks on insert
          and compound-based lookups (leading column must be drug_id).

    Args:
        conn: An open psycopg2 connection to ``pharmacodb``.
    """
    cur = conn.cursor()
    # Covering index for the D2.002 query — all columns needed are included so
    # Postgres can satisfy the query entirely from the index (index-only scan),
    # never touching the heap. Order: equality filter first, range filter second,
    # remaining selected columns after.
    cur.execute(
        'CREATE INDEX ON gene_drugs ("mDataType", pvalue, gene_id, drug_id, estimate)'
    )
    # Separate index for compound-based lookups and FK constraint checks on insert.
    cur.execute("CREATE INDEX ON gene_drugs (drug_id)")
    conn.commit()
    log.info("Indexes created")

    # VACUUM ANALYZE updates the visibility map so Postgres knows all pages are
    # all-visible, enabling index-only scans on the covering index. Must run
    # outside a transaction block (isolation_level=0).
    conn.set_isolation_level(0)
    cur.execute("VACUUM ANALYZE gene_drugs")
    log.info("VACUUM ANALYZE complete")


def _fetch_compound_detail(compound_id):
    """Fetch SMILES and PubChem CID for a single compound from the GraphQL API.

    The bulk ``compounds(all: true)`` endpoint does not return SMILES; the
    individual ``compound(compoundId: X)`` endpoint does. Empty strings are
    normalised to ``None``.

    All compounds are returned regardless of whether smiles/pubchem are
    available, so that ``drug_annots`` contains every compound ID and
    ``gene_drugs`` foreign-key constraints are always satisfied. The downstream
    ``parser.py`` query (``WHERE smiles IS NOT NULL OR pubchem IS NOT NULL``)
    handles its own filtering.

    Args:
        compound_id (int): PharmacoDB compound ID.

    Returns:
        tuple: ``(drug_id, smiles, pubchem)`` where smiles/pubchem may be None.
    """
    data = _gql("""
    { compound(compoundId: %d) {
        compound { id annotation { smiles pubchem } }
    } }
    """ % compound_id)
    c = data["compound"]["compound"]
    ann = c.get("annotation") or {}
    smiles = ann.get("smiles") or None
    pubchem = ann.get("pubchem") or None
    if smiles == "":
        smiles = None
    if smiles is None and pubchem is None:
        return None
    return (c["id"], smiles, pubchem)


def _populate_drug_annots(conn):
    """Fetch all compounds from the API and populate the ``drug_annots`` table.

    Two-step process:
        1. Bulk query ``compounds(all: true)`` to retrieve all compound IDs.
        2. Individual ``compound(compoundId: X)`` queries (parallelised with
           ``MAX_WORKERS`` threads) to retrieve SMILES and PubChem CID, which
           are not available in the bulk endpoint.

    Args:
        conn: An open psycopg2 connection to ``pharmacodb``.

    Returns:
        set[int]: Set of all drug IDs inserted, used downstream to enforce
        foreign-key safety when populating ``gene_drugs``.
    """
    log.info("Fetching all compound IDs...")
    data = _gql("{ compounds(all: true) { id } }")
    compound_ids = [c["id"] for c in data["compounds"]]
    log.info("Found %d compounds, fetching details...", len(compound_ids))

    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_compound_detail, cid): cid
                   for cid in compound_ids}
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result is not None:
                rows.append(result)
            if done % 1000 == 0:
                log.info("  compounds processed: %d / %d", done, len(compound_ids))

    cur = conn.cursor()
    for i in range(0, len(rows), BATCH_SIZE):
        cur.executemany(
            "INSERT INTO drug_annots (drug_id, smiles, pubchem) VALUES (%s, %s, %s)",
            rows[i:i + BATCH_SIZE]
        )
        conn.commit()
    log.info("Inserted %d rows into drug_annots", len(rows))
    return {r[0] for r in rows}


def _populate_genes(conn):
    """Fetch all genes from the API and populate the ``genes`` table.

    A single bulk query ``genes(all: true)`` retrieves all 61K genes.
    ``gene_name`` is set to ``annotation.symbol`` (HGNC gene symbol, e.g.
    ``TP53``) when available, falling back to the raw ``name`` field otherwise.
    The fallback covers two cases:

    - Fusion genes such as ``BCR_ABL``, which have no HGNC symbol because
      they are not canonical genome loci.
    - ~593 Ensembl entries (e.g. ``ENSG00000069712``) that lack an approved
      symbol; these are uncharacterised loci unlikely to pass the D2.002
      pvalue filter in practice.

    Args:
        conn: An open psycopg2 connection to ``pharmacodb``.

    Returns:
        list[int]: Ordered list of all gene IDs inserted.
    """
    log.info("Fetching all genes...")
    data = _gql("{ genes(all: true) { id name annotation { symbol } } }")
    genes = data["genes"]

    rows = []
    for g in genes:
        ann = g.get("annotation") or {}
        gene_name = ann.get("symbol") or g["name"]
        rows.append((g["id"], gene_name))

    cur = conn.cursor()
    for i in range(0, len(rows), BATCH_SIZE):
        cur.executemany(
            "INSERT INTO genes (gene_id, gene_name) VALUES (%s, %s)",
            rows[i:i + BATCH_SIZE]
        )
        conn.commit()
    log.info("Inserted %d rows into genes", len(rows))
    return [r[0] for r in rows]


def _fetch_gene_drugs(gene_id):
    """Fetch all drug-gene correlations for a single gene.

    Uses ``gene_compound_dataset(geneId: X, all: true)`` — the ``all: true``
    flag is required; without it the API returns only the first 20 rows,
    silently truncating most compounds for any gene with broad coverage.

    Args:
        gene_id (int): PharmacoDB gene ID.

    Returns:
        tuple[int, list[dict]]: ``(gene_id, records)`` where each record
        contains ``compound.id``, ``estimate``, ``pvalue_analytic``, and
        ``mDataType``.
    """
    data = _gql("""
    { gene_compound_dataset(geneId: %d, all: true) {
        compound { id }
        estimate
        pvalue_analytic
        mDataType
    } }
    """ % gene_id)
    return gene_id, data["gene_compound_dataset"]


def _map_mdatatype(mdt):
    """Normalise new API mDataType values to the legacy ``'mRNA'`` label.

    The old PharmacoDB SQL database stored all mRNA expression data under the
    single label ``'mRNA'``. The new API distinguishes quantification pipelines:

    - ``'Kallisto_0.46.1.rnaseq'`` → ``'mRNA'`` (RNA-seq via Kallisto)
    - ``'rna'``                     → ``'mRNA'`` (microarray / older pipeline)
    - ``'cnv'``, ``'mutation'``     → kept as-is (non-expression data types)

    This mapping ensures the D2.002 filter ``mDataType = 'mRNA'`` continues to
    work without modifying any downstream scripts.

    Args:
        mdt (str): mDataType value from the GraphQL API.

    Returns:
        str: Normalised mDataType for storage in ``gene_drugs``.
    """
    if mdt in MRNA_TYPES:
        return "mRNA"
    return mdt


def _populate_gene_drugs(conn, gene_ids, valid_drug_ids):
    """Fetch all drug-gene correlations and populate the ``gene_drugs`` table.

    Iterates over all gene IDs (parallelised with ``MAX_WORKERS`` threads),
    querying ``gene_compound_dataset(geneId: X, all: true)`` for each. Rows
    are batch-inserted every ``BATCH_SIZE`` records to avoid holding large
    result sets in memory.

    Mapping applied on insert:
        - ``pvalue_analytic`` stored as ``pvalue`` (``pvalue_permutation`` is
          null across the entire new API dataset).
        - ``mDataType`` normalised via :func:`_map_mdatatype`.

    Rows whose ``compound.id`` is not in ``valid_drug_ids`` are skipped to
    preserve foreign-key integrity with ``drug_annots``.

    Args:
        conn: An open psycopg2 connection to ``pharmacodb``.
        gene_ids (list[int]): All gene IDs to iterate over.
        valid_drug_ids (set[int]): Set of drug IDs present in ``drug_annots``.
    """
    log.info("Fetching gene-drug correlations for %d genes...", len(gene_ids))
    total_rows = 0
    done = 0

    cur = conn.cursor()
    batch = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_gene_drugs, gid): gid for gid in gene_ids}
        for future in as_completed(futures):
            done += 1
            gene_id, records = future.result()
            for r in records:
                drug_id = r["compound"]["id"]
                if drug_id not in valid_drug_ids:
                    continue
                batch.append((
                    drug_id,
                    gene_id,
                    r["estimate"],
                    r["pvalue_analytic"],
                    _map_mdatatype(r["mDataType"])
                ))
            if len(batch) >= BATCH_SIZE:
                cur.executemany(
                    """INSERT INTO gene_drugs
                       (drug_id, gene_id, estimate, pvalue, "mDataType")
                       VALUES (%s, %s, %s, %s, %s)""",
                    batch
                )
                conn.commit()
                total_rows += len(batch)
                batch = []
            if done % 1000 == 0:
                log.info("  genes processed: %d / %d  (rows so far: %d)",
                         done, len(gene_ids), total_rows)

    if batch:
        cur.executemany(
            """INSERT INTO gene_drugs
               (drug_id, gene_id, estimate, pvalue, "mDataType")
               VALUES (%s, %s, %s, %s, %s)""",
            batch
        )
        conn.commit()
        total_rows += len(batch)

    log.info("Inserted %d rows into gene_drugs", total_rows)


def fetch_pharmacodb():
    """Fetch PharmacoDB GraphQL data and populate the local pharmacodb DB."""
    from chemicalchecker.util import Config
    config = Config()

    _ensure_db(config)
    conn = _connect(config)
    try:
        _create_tables(conn)
        valid_drug_ids = _populate_drug_annots(conn)
        gene_ids = _populate_genes(conn)
        _populate_gene_drugs(conn, gene_ids, valid_drug_ids)
        _create_indexes(conn)
    finally:
        conn.close()

    # Create the datasource data_path directory so Datasource.available returns
    # True and Datasource.test_all_downloaded() does not flag pharmacodb as missing.
    data_path = os.path.join(config.PATH.CC_DATA, DB_NAME)
    os.makedirs(data_path, exist_ok=True)
    log.info("fetch_pharmacodb complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Fetch PharmacoDB from GraphQL API into PostgreSQL"
    )
    parser.add_argument("-c", "--config", help="Path to cc_config.json")
    args = parser.parse_args()
    if args.config:
        os.environ["CC_CONFIG"] = args.config
    fetch_pharmacodb()
