# Custom BLASTn Databases for Taxonomic Validation
This guide explains how to prepare a custom BLAST database and its corresponding taxonomy file for taxonomic validation within the BeeGees pipeline. 

## Overview
The taxonomic validation step (run via `tv_local_blast.py` and `tv_blast2taxonomy.py`) requires three components configured in your `config.yaml`:

1. A **BLAST database** - specified via `taxonomic_validation.database`
2. A **taxonomy TSV file** - specified via `taxonomic_validation.database_taxonomy`
3. An **input/expected taxonomy file** - specified via `expected_taxonomy`

These paths are set in the config file and passed to the validation scripts automatically by Snakemake.

---

## Config.yaml Setup
In your `config.yaml`, the taxonomic validation section should point to your custom database and taxonomy file:

```yaml
## Taxonomic validation
taxonomic_validation:
    # Path to BLAST database directory or FASTA file
    database: "/path/to/your/blast_database/"
    
    # Path to taxonomy TSV file (see formats below)
    database_taxonomy: "/path/to/your/taxonomy.tsv"
    
    # Expected taxonomy for your samples (must contain Process ID column)
    expected_taxonomy: "/path/to/your/samples_metadata.csv"
```

---

## Taxonomy TSV File Formats
The taxonomy file maps each sequence ID in your BLAST database to its taxonomic classification. The script auto-detects and accepts two formats:

### Format A: Generic (Semicolon-Delimited)
A two-column TSV with `Feature ID` and `Taxon` headers:

| Column | Description |
|--------|-------------|
| `Feature ID` | Sequence identifier (accession number/sseqid) |
| `Taxon` | Semicolon-delimited lineage with rank prefix codes |

**E.g.:**
```
Feature ID	Taxon
MH021436.1	k__Viridiplantae;p__Streptophyta;c__Magnoliopsida;o__Solanales;f__Solanaceae;g__Solanum;s__Solanum bukasovii
MH522400.1	k__Viridiplantae;p__Streptophyta;c__Magnoliopsida;o__Poales;f__Poaceae;g__Melocanna;s__Melocanna baccifera
```

**Prefix codes:**
| Prefix | Rank |
|--------|------|
| `k__` | Kingdom |
| `p__` | Phylum |
| `c__` | Class |
| `o__` | Order |
| `f__` | Family |
| `g__` | Genus |
| `s__` | Species |

### Format B: BOLDistilled-Style (Multi-Column)
A TSV with separate columns for each taxonomic rank:

| Column | Description |
|--------|-------------|
| `bin` | Sequence identifier (accession number/sseqid) |
| `kingdom`, `phylum`, `class`, `order`, `family`, `genus`, `species` | Tab-separated taxonomic lineage |

**E.g.:**
```
bin	kingdom	phylum	class	order	family	genus	species
MH021436.1	Viridiplantae	Streptophyta	Magnoliopsida	Solanales	Solanaceae	Solanum	Solanum bukasovii
MH522400.1	Viridiplantae	Streptophyta	Magnoliopsida	Poales	Poaceae	Melocanna	Melocanna baccifera
```

Optional additional columns (`subfamily`, `tribe`, `subspecies`) are also supported.

### Important Notes
- The taxonomy file must be **tab-delimited** (TSV)
- The sequence ID in the taxonomy file must exactly match what BLAST returns in the hit ID (the sseqid)
- Empty ranks are permitted (leave the field blank (format B) or omit from the semicolon string (format A))

---

## Creating a BLAST Database
The `taxonomic_validation.database` config parameter can point to either:
- An existing BLAST database directory or a FASTA file (the pipeline will create the database automatically)

### From a Multi-FASTA File
If you have a multi-FASTA file of reference sequences and want to pre-build the database:

```bash
makeblastdb -in rbcl_seqs.fasta \
            -dbtype nucl \
            -out my_custom_rbcl_db \
            -parse_seqids
```

The `-parse_seqids` flag is necessary to allow BLAST to return the sequence identifiers (sseqids) in the BLAST output, as these link to the IDs in the taxonomy TSV file.

## Generating a Taxonomy TSV (format A)
If you have an existing BLAST database, you can extract the accessions and generate a taxonomy file (format A) using `blastdbcmd` and [TaxonKit](https://bioinf.shenwei.me/taxonkit/).

### Step 1: Install and Set Up TaxonKit
```bash
conda install -c bioconda taxonkit
```
Follow the [TaxonKit setup instructions](https://bioinf.shenwei.me/taxonkit/usage/#before-use) to download the required NCBI taxdump files.

### Step 2: Extract Accessions and Taxids from BLAST Database
```bash
blastdbcmd -db core_nt_rbcL \
           -entry all \
           -outfmt "%a\t%T" \
           > accession_taxid.tsv
```

This outputs accessions and numeric taxids, however the script requires full taxonomic lineage information.

### Step 3: Convert Taxids to Lineage Strings (format A)
```bash
cut -f2 accession_taxid.tsv | \
    taxonkit lineage | \
    taxonkit reformat -f "k__{k};p__{p};c__{c};o__{o};f__{f};g__{g};s__{s}" \
    > taxid_lineage.tsv
```

### Step 4: Join Files into Final Taxonomy TSV (format A)
```bash
paste <(cut -f1 accession_taxid.tsv) <(cut -f3 taxid_lineage.tsv) | \
    sed '1i Feature ID\tTaxon' \
    > rbcl_taxonomy.tsv
```

This produces a generic format taxonomy file ready for use in the pipeline.

## Expected Taxonomy File (samples.csv)
The `taxonomic_validation.expected_taxonomy` config parameter points to a file containing the expected taxonomy for each of your samples. The file must have either a `Process ID` or `ID` column to link samples to sequences, plus taxonomic rank columns (`phylum`, `class`, `order`, `family`, `genus`, `species`), such as the ``samples.csv example format (hierarchical taxonomy)` mentioned in the [set up](https://github.com/bge-barcoding/BeeGees?tab=readme-ov-file#generate-input-sample-csv-file) section of the repository


## ID Matching
The script extracts sequence IDs from BLAST hits as follows:

- If the hit ID contains pipes (e.g., `gi|12345|gb|MH021436.1|`), the last element after splitting on `|` is used
- Otherwise, the hit ID is used directly

Ensure your taxonomy file uses IDs that match this extracted format.

_If you also have a HMM corresponding to the same marker/barcode, the BeeGees pipeline can structurally and taxonomically validate those recovered barcodes. If you create a custom BLAStn database, taxonomy mapping file, and HMM (other than for COI and rbcL), please consider kindly creating a pull request so expand the BeeGees pipeline's capabilities._
