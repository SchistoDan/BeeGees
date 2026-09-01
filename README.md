# BeeGees - Barcode gene Extraction and Evaluation from Genome Skims #

[![Snakemake](https://img.shields.io/badge/snakemake-9.9.0-brightgreen.svg?style=flat)](https://snakemake.readthedocs.io)
[![Molecular Ecology Resources](https://img.shields.io/badge/publication_DOI-10.1111%2F1755--0998.70170-blue.svg)](https://doi.org/10.1111/1755-0998.70170)
[![CI](https://github.com/bge-barcoding/BeeGees/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bge-barcoding/BeeGees/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/beegees.svg?icon=si%3Apython)](https://badge.fury.io/py/beegees)
[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/beegees/README.html)


BeeGees is a Snakemake workflow for recovering high-quality protein-coding DNA barcodes from low-coverage NGS data at scale. Built around [MitoGeneExtractor](https://github.com/cmayer/MitoGeneExtractor) and tailored for genome skims of museum specimens, it takes raw reads through preprocessing, reference retrieval, barcode recovery, consensus cleaning, and structural and taxonomic validation, producing a validated barcode FASTA, a unified per-sample metrics CSV, and an interactive MultiQC report.

> **Supported markers**: **COI-5P** and **rbcL** (so far).

> **Supported sequence data**: paired-end Illumina/Element Biosciences & single-end Ultima Genomics.


---


# Contents #
- [Requirements](#requirements)
- [Installation](#Installation)
- [Quick start](#quick-start)
- [Workflow](#workflow)
- [Preparing inputs](#preparing-inputs)
  - [samples.csv](#samplescsv)
  - [Run mode detection](#run-mode-detection)
  - [Sample-specific pseudo-reference retrieval](#sample-specific-pseudo-reference-retrieval)
- [Configuration](#configuration)
  - [config.yaml](#configyaml)
  - [Cluster profiles](#cluster-profiles)
- [Results structure](#results-structure)
- [Validation process](#validation-process)
  - [Structural validation](#structural-validation)
  - [Taxonomic validation](#taxonomic-validation)
- [Major outputs](#major-outputs)
  - [Validated barcode FASTA](#validated-barcode-fasta)
  - [Final metrics CSV](#final-metrics-csv)
  - [Diagnostic plots](#diagnostic-plots)
  - [MultiQC report](#multiqc-report)
- [Screening negative controls](#screening-negative-controls)
- [Citations and contributions](#citations-and-contributions)
- [Future developments](#future-developments)


---

 
# Requirements #
- **Reads** in `.fastq`/`.fastq.gz` format - either paired-end or single-end from Ultima Genomics sequencing.
- **`samples.csv`**
- **`sequence_references.csv`**
- **Activated conda environment** (see `beegees_env.yaml`).
> **Note for pip users:** all 'scientific tool' dependencies (Snakemake, MitoGeneExtractor, fastp, TrimGalore, BLAST, nhmmer, etc.) come from the BeeGees conda environment. `pip install beegees` provides the CLI only - the pipeline will not run without the BeeGees conda environment active.


---


# Installation # 
The `beegees` pipeline can currently be downloaded via by:
1. Cloning the beegees respository
```
gh repo clone bge-barcoding/BeeGees
# Or
wget https://github.com/bge-barcoding/BeeGees/archive/refs/heads/main.zip
```
3. As a pip through PyPi
```
pip install beegees
```
> **Pip users:** after installation, run `beegees init` in your working directory to copy the template `config/`, `profiles/`, and `samples_template.csv` into the current directory. These files are required before running the pipeline (see section below).

> `beegees` will be available as a conda package through bioconda in the near future.


--- 


# Quick start #

> **Installation note:** `pip install beegees` provides the `beegees` CLI only — all scientific tool dependencies (Snakemake, MitoGeneExtractor, fastp, TrimGalore, BLAST, nhmmer, etc.) must come from the conda environment. A bioconda package that bundles all dependencies is planned; until then, the clone workflow below is recommended for all users.

### 1. Clone the repository and create the conda environment
```bash
git clone https://github.com/bge-barcoding/BeeGees.git
cd BeeGees
conda env create -f beegees_env.yaml
conda activate beegees_env
```

### 2. Complete samples.csv
Create `samples.csv` or edit `beegees/config/samples_template.csv`, with sample identifier (ID), forward (SE data only) and reverse read paths (PE data), and hierarchical taxonomy or NCBI taxonomic identifiers (taxid) — [see section below for details](#samplescsv).

### 3. Download a BLASTn database and taxonomy mappings file
Required for taxonomic validation of generated barcode consensus sequences — [see section below for details on available databases](#validation-process) or [guidance on creating your own BLASTn database and mapping file](https://github.com/bge-barcoding/BeeGees/blob/main/docs/README_custom_blast_dbs.md).

### 4. Populate config.yaml
Fill in paths to required files, set parameters, credentials, and resource allocations — [see section below for details](#configyaml).

### 5. Run BeeGees using the `beegees run` command with the appropriate profile:

**SLURM** (recommended for SLURM-based HPC clusters): run directly on the login node — Snakemake farms each rule out as a separate SLURM job:
```bash
sbatch run_slurm.sh
```

**Local** (all rules run on the current node — suitable for interactive compute sessions):
```bash
# Create an interactive SLURM session
srun --pty --mem=128G --cpus-per-task=16 --time=08:00:00 bash

# Run BeeGees locally
bash run_local.sh
```

> Use `beegees run --help` for additional options (e.g. `--cores`, `--dryrun`, `--log-file PATH` to write a live log file).
> Depending on your HPC cluster architecture and job scheduler, you may need to edit the partition in `run_*.sh`, `config.yaml` rule resource block, and `slurm_partition` in `profiles/slurm/config.yaml`.


# Workflow #
<img width="1829" height="853" alt="BeeGees workflow diagram" src="https://github.com/user-attachments/assets/017062f4-51c1-43b3-b9fd-6277820b45e4" />


1. **Preprocessing** - the run mode is detected automatically from `samples.csv` (see [Run mode detection](#run-mode-detection)):
   - **PE concat**: adapter trimming, quality filtering, poly-G trimming and deduplication with [fastp](https://github.com/OpenGene/fastp), then R1+R2 concatenation, secondary quality trimming with [TrimGalore](https://github.com/FelixKrueger/TrimGalore), and optional downsampling.
   - **PE merge**: quality control and merging of overlapping read pairs with fastp, header cleaning for MitoGeneExtractor compatibility, and optional downsampling.
   - **SE (Ultima Genomics)**: adapter trimming, poly-X tail trimming and deduplication with fastp using Ultima-specific settings. No merge or concat step.
2. **Sample-specific reference retrieval** - taxonomically appropriate protein references are pulled from GenBank by [Gene Fetch](https://github.com/bge-barcoding/gene_fetch).
3. **Barcode recovery** - protein reference-guided extraction of barcode sequences from preprocessed reads using MitoGeneExtractor, producing initial consensus sequences.
4. **Consensus preparation** - header standardisation and concatenation of raw consensus sequences into multi-FASTA.
5. **Consensus cleaning and filtering (`fasta_cleaner`)** - sequential filters applied to MGE read alignments to remove contaminants and outliers before generating cleaned consensus sequences:
   - Human COI contamination removal, common in museum specimens ([`01_human_cox1_filter.py`](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/01_human_cox1_filter.py))
   - AT content filtering, targeting suspected fungal/bacterial contamination ([`02_at_content_filter.py`](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/02_at_content_filter.py))
   - Statistical outlier removal of reads dissimilar to the initial consensus ([`03_statistical_outlier_filter.py`](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/03_statistical_outlier_filter.py))
   - Optional custom reference-based filtering ([`04_reference_filter.py`](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/04_reference_filter.py))
   - Cleaned consensus generation and metrics aggregation ([`05_consensus_generator.py`](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/05_consensus_generator.py))
6. **Barcode validation and selection** (see [Validation process](#validation-process)):
   - **Structural validation** - HMM-based barcode extraction, reading frame analysis, stop codon detection and quality ranking ([`structural_validation.py`](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/structural_validation.py))
   - **Local BLASTn search** - chunked, parallel searches of structurally validated barcodes against a local reference database ([`tv_local_blast.py`](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/tv_local_blast.py))
   - **Taxonomic validation** - hierarchical matching of BLAST results against expected taxonomy, selecting the best sequence per sample ([`tv_blast2taxonomy.py`](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/tv_blast2taxonomy.py))
7. **Statistics compilation** - QC, recovery, cleaning, filtering and validation metrics aggregated into CSV reports ([`compile_barcoding_stats.py`](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/compile_barcoding_stats.py)).
8. **Final integration** - all pipeline metrics merged into a unified output CSV ([`val_csv_merger.py`](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/val_csv_merger.py)).
9. **Barcoding outcome** - per-sample success called as PASS/PARTIAL/FAIL from the unified CSV ([`barcoding_outcome.py`](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/barcoding_outcome.py)).
10. **Cleanup** - removal of temporary files and redundant sample-specific logs.

---


# Preparing inputs #
 
## samples.csv ##
- This can be created manually, or via the [sample-processing](https://github.com/bge-barcoding/sample-processing) workflow if working from specimen metadata downloaded from the Barcode of Life Data System (BOLD).
- Must contain the headers **`ID`**, **`forward`**, and either **`taxid`** _or_ hierarchical taxonomy columns (`phylum`–>`species`).

- **`ID`** - unique sample identifier. Because of regex matching and statistics aggregation, the sample ID is taken as the string before the first underscore, so **avoid `_` characters in sample names** (e.g. `BSNHM002-24` rather than `BSNHM002_24`, `P3-1-A10-2-G1` rather than `P3_1_A10_2_G1`).
- **`forward`** - absolute path to the forward (R1) read file, gzipped or not. For SE/Ultima runs this is the only read file required.
- **`reverse`** _(optional)_ - absolute path to the reverse (R2) read file. See [Run mode detection](#run-mode-detection).
- **`taxid`** _or_ **hierarchical taxonomy** - either an NCBI taxid, found by searching the expected species/genus/family in the [NCBI taxonomy database](https://www.ncbi.nlm.nih.gov/taxonomy), or a taxonomic lineage (columns `phylum`, `class`, `order`, `family`, `genus`, `species`), from which the taxid of the lowest identified rank is retrieved.


**Paired-end (taxid)**
 
| ID | forward | reverse | taxid |
| --- | --- | --- | --- |
| BSNHM002-24 | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | 177658 |
| BSNHM038-24 | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | 177627 |
| BSNHM046-24 | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | 3084599 |
 
**Single-end / Ultima (taxid)**
 
| ID | forward | reverse | taxid |
| --- | --- | --- | --- |
| UK001-A01 | abs/path/to/reads.fq.gz | | 177658 |
| UK001-A02 | abs/path/to/reads.fq.gz | | 177627 |
| UK001-A03 | abs/path/to/reads.fq.gz | | 3084599 |
 
**Paired-end (hierarchical taxonomy)**
 
| ID | forward | reverse | phylum | class | order | family | genus | species |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BSNHM002-24 | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | Arthropoda | Insecta | Hemiptera | Cicadidae | Tibicina | Tibicina tomentosa |
| BSNHM038-24 | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | Tracheophyta | Pinopsida | Pinales | Pinaceae | Abies | |
| BSNHM046-24 | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | Annelida | Polychaeta | Terebellida | Ampharetidae | Samytha | Samytha sexcirrata |


---
 

## Run mode detection ##
Run mode is inferred from the `reverse` column of `samples.csv`:
- **Paired-end** - a `reverse` path is provided for all samples. Both `concat` and `merge` preprocessing modes run.
- **Single-end (Ultima)** - the `reverse` column is absent or empty for all samples. Only `se` mode runs.
**The `reverse` column must be consistent across all samples.** Mixed SE/PE samples in the same CSV cause the pipeline to exit with an error.
 
> **SE/Ultima preprocessing specifics** - fastp runs in SE mode with Ultima-tuned settings: poly-X tail trimming (targeting the poly-T 3' artefacts characteristic of Ultima flow-based chemistry), poly-G trimming disabled (Ultima is not a two-colour system), and deduplication enabled. Adapter auto-detection is the primary adapter trimming mechanism, on the assumption that Ultima-specific adapters have already been removed by Ultima Genomics' Trimmer tool. Barcode recovery and all downstream processing are identical to PE mode, with outputs written to `se_mode/` (analogous to `merge_mode/` and `concat_mode/`).


---


## Sample-specific pseudo-reference retrieval ##
`sequence_references.csv` can be created manually or generated within the workflow by [Gene Fetch](https://github.com/bge-barcoding/gene_fetch) (recommended). Setting `run_gene_fetch: true` in `config.yaml` retrieves protein pseudo-references for each sample from NCBI GenBank using its taxid or taxonomic hierarchy. This requires a sequence target (e.g. COI) and your NCBI credentials - email address and API key (see [guidance on obtaining a key](https://support.nlm.nih.gov/kbArticle/?pn=KA-05317)).
 
The file must contain the headers **`ID`** and **`protein_reference_path`**:
- **`ID`** - must match the `ID` column of `samples.csv` exactly.
- **`protein_reference_path`** - absolute path to the protein pseudo-reference used for sample-specific protein-guided read alignment.

| ID | protein_reference_path |
| --- | --- |
| BSNHM002-24 | path/to/BSNHM002-24.fasta |
| BSNHM038-24 | path/to/BSNHM038-24.fasta |
| BSNHM046-24 | path/to/BSNHM046-24.fasta |
 
> **Important:** the sample ID, the reference FASTA filename, and the reference FASTA header **must** currently all be identical for correct sample-to-reference mapping. Gene Fetch handles this automatically.


---


# Configuration #
 
## config.yaml ##
- Update `beegees/config/config.yaml` (generated by `beegees init` into the current directory for pip users) with the required paths, variables, and credentials.
- 
**General**
```
run_name: BeeGees run identifier
samples_file: Path to samples.csv
sequence_reference_file: Path to sequence_references.csv (leave blank if run_gene_fetch == true)
output_dir: Path to output directory (created if it does not exist)
```

**Preprocessing (QC)**
```
adapter_r1: PE R1 adapter sequence for fastp (empty = Illumina TruSeq default)
adapter_r2: PE R2 adapter sequence for fastp (empty = Illumina TruSeq default)
extra_fastp_args: Any additional fastp flags, as a single quoted string
```

**Gene Fetch** ([Gene Fetch repository](https://github.com/SchistoDan/gene_fetch))
```
run_gene_fetch: Use Gene Fetch to generate reference sequences (default: true)
email: Email for NCBI API. Required if run_gene_fetch == true
api_key: NCBI API key. Required if run_gene_fetch == true
gene: Target gene (cox1 or rbcl)
type: Sequence type to fetch: "protein" or "both" (protein + nucleotide) (default: both)
minimum_length: Minimum protein pseudo-reference length in amino acids (default: 500)
input_type: Taxonomic identification column(s) in samples.csv - taxid or hierarchical (default: taxid)
genbank: Download complete GenBank records of retrieved protein pseudo-references
```

**Downsampling**
```
enabled: Enable downsampling (default: false)
max_reads: Maximum reads/read-pairs to subsample to, passed to reformat.sh. Downsampling occurs post-QC.
           SE reads = number of individual reads
           PE reads = number of read PAIRS 
           max_reads: 10000000 gives 10M SE reads, or 10M pairs (20M individual reads for PE). 
           Set to 0 to disable
```

**MitoGeneExtractor** ([command-line options](https://github.com/cmayer/MitoGeneExtractor/tree/main?tab=readme-ov-file#command-line-options))
```
r: Exonerate relative score threshold(s)
s: Exonerate minimum score threshold(s)
n: Base pairs to extend beyond the Exonerate alignment (default: 0)
C: Genetic code for Exonerate (https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi)
t: Consensus threshold (e.g. 0.5 = 50%) (default: 0.5)
```

**fasta_cleaner** - applied in order: (01) human COI → (02) AT content → (03) statistical outlier → (04, optional) reference-based → (05) cleaned consensus generation → (06) metrics aggregation
```
consensus_threshold: Proportion of bases at each position that must agree to be included in the consensus (e.g. 0.5 = ≥50%)
human_threshold: Similarity to the human COI reference above which reads are removed (e.g. 0.95)
at_difference: AT content deviation from the consensus above which reads are removed (e.g. 0.1 = 10%)
at_mode: Absolute (remove if AT content differs by more than the threshold in either direction), Higher (remove only above threshold), or Lower (remove only below threshold)
outlier_percentile: Similarity to the consensus below which reads are flagged as statistical outliers and removed (e.g. 90.0)
reference_dir: Directory containing at least one [ID].fasta of known contaminant or target species genome(s), depending on reference_filter_mode
reference_filter_mode: keep_similar (reference-based retention) or remove_similar (contaminant removal)
```

**Structural validation**
```
target: Barcode marker to extract, corresponding to HMM files in resources/hmm (cox1 or rbcl)
verbose: Enable verbose logging (default: false)
```

**Taxonomic validation**
```
database: Path to a BLASTn database directory, or to a FASTA file to build one from (using makeblastdb)
database_taxonomy: TSV of taxonomic mappings corresponding to records in the BLASTn database
taxval_rank: Highest taxonomic rank to validate at (default and recommended: family)
expected_taxonomy: CSV with columns Process ID,phylum,class,order,family,genus,species, where
                   Process ID equals ID in samples_file. If hierarchical taxonomy was supplied in
                   samples.csv, that file can be reused here
min_pident: Minimum percent identity for a BLAST hit to be retained
min_length: Minimum alignment length for a BLAST hit to be retained
verbose: Enable verbose logging (default: true)
```

**Resource allocation**
```
rules: Each main rule specifies requested threads and memory (Mb), with dynamic memory scaling on
       retry (mem_mb * retry #). Remember to set PARTITION for the Gene Fetch, MitoGeneExtractor,
       structural_validation and taxonomic_validation rules
```


---


## Cluster profiles ##
The `profiles/` directory contains `config.yaml` files for SLURM and local submission. Pip users can generate this directory with `beegees init`; for cloned repositories it is at `beegees/profiles/`. Beyond `slurm_partition` and `jobs` (if running `/profiles/slurm/config.yaml`), the defaults can usually be left alone.
 
- **`slurm_partition`** - the default partition for each Snakemake job, unless overridden in `config/config.yaml` rule resource block. Use a partition with at least a 6-12 hour time limit to be safe.
- **`jobs`** (SLURM profile only) - maximum number of concurrent workflow jobs. This is a fair-share/queue-etiquette limit rather than a local resource one - the jobs themselves run on compute nodes. Raise for large batches if the queue allows. Too low creates a bottleneck; too high risks hitting filesystem limits, job submission limits, quotas and fairshare policies, leaving many jobs pending. For example, under a 256 GB per-user memory limit, `jobs: 20` with 32 GB per MitoGeneExtractor job means only 8 run in parallel while the remaining 12 'pend'.

>Select the profile with the `--profile` flag of `beegees run`.


---


# Results structure #
 
**Key outputs at the top level of `output_dir/`:**
 
| File | Description |
| --- | --- |
| `{run_name}_validated_barcodes.fasta` | Final validated barcodes (only if both validation steps run) |
| `{run_name}_final_metrics.csv` | Unified per-sample metrics across the whole pipeline |
| `multiqc_report.html` | Interactive summary report (copy of the one in `05_barcoding_outcome/`) |
 
<details>
<summary><b>Full directory tree</b></summary>

```
output_dir/
├── 01_preprocessing/
│   ├── merge_mode/                                            # PE mode only
│   │   ├── trimmed_data/
│   │   │   └── {sample}/
│   │   │       ├── {sample}_merged.fastq.gz                   # Merged paired-end reads
│   │   │       ├── {sample}_merged_clean.fastq(.gz)           # Header-cleaned merged reads
│   │   │       ├── {sample}_fastp_report.html
│   │   │       └── {sample}_fastp_report.json
│   │   ├── logs/
│   │   │   ├── clean_headers/clean_headers.log                # Aggregated header cleaning logs
│   │   │   ├── fastp/                                         # Per-sample fastp logs
│   │   │   └── final_cleanup_complete.txt
│   │   └── fastp_summary-merge.csv                            # Per-sample fastp QC summary
│   ├── concat_mode/                                           # PE mode only
│   │   ├── trimmed_data/
│   │   │   └── {sample}/
│   │   │       ├── {sample}_R1_trimmed.fastq.gz
│   │   │       ├── {sample}_R2_trimmed.fastq.gz
│   │   │       ├── {sample}_concat_trimmed.fq                 # Quality-trimmed concatenated reads
│   │   │       ├── {sample}_fastp_report.html
│   │   │       ├── {sample}_fastp_report.json
│   │   │       └── {sample}_concat.fastq_trimming_report.txt  # TrimGalore report
│   │   ├── logs/
│   │   │   ├── concat/concat_reads.log
│   │   │   ├── trim_galore/trim_galore.log
│   │   │   ├── fastp/                                         # Per-sample fastp logs
│   │   │   ├── gzip/                                          # Per-sample compression logs
│   │   │   └── final_cleanup_complete.txt
│   │   └── fastp_summary-concat.csv
│   └── se_mode/                                               # SE/Ultima mode only
│       ├── trimmed_data/
│       │   └── {sample}/
│       │       ├── {sample}_se_trimmed.fastq                  # Trimmed SE reads
│       │       ├── {sample}_fastp_report.html
│       │       └── {sample}_fastp_report.json
│       ├── logs/
│       │   ├── fastp/
│       │   └── final_cleanup_complete.txt
│       └── fastp_summary-se.csv
│
├── 02_references/                                             # Only if run_gene_fetch = true
│   ├── protein/{sample}.fasta                                 # Per-sample protein references
│   ├── genbank/                                               # GenBank records (if genbank: true)
│   ├── sequence_references.csv                                # Reference metadata
│   └── gene_fetch.log
│
├── 03_barcode_recovery/
│   ├── merge_mode/                                            # PE mode only
│   │   ├── alignment/
│   │   │   └── {sample}_r_{r}_s_{s}_align_{sample}.fas        # MGE alignment files
│   │   ├── consensus/
│   │   │   ├── {sample}_r_{r}_s_{s}_con_{sample}.fas          # Individual consensus files
│   │   │   └── {run_name}_cons_combined-merge.fasta           # Combined consensus sequences
│   │   ├── fasta_cleaner/
│   │   │   ├── 01_human_filtered/
│   │   │   │   ├── human_filtered.txt
│   │   │   │   └── human_filter_metrics.csv
│   │   │   ├── 02_at_filtered/
│   │   │   │   ├── at_filtered.txt
│   │   │   │   └── at_filter_summary.csv
│   │   │   ├── 03_outlier_filtered/
│   │   │   │   ├── outlier_filtered.txt
│   │   │   │   ├── outlier_filter_summary_metrics.csv
│   │   │   │   └── outlier_filter_individual_metrics.csv
│   │   │   ├── 04_reference_filtered/                         # Optional
│   │   │   │   ├── reference_filtered.txt
│   │   │   │   └── reference_filter_metrics.csv
│   │   │   ├── 05_cleaned_consensus/
│   │   │   │   └── cleaned_cons_metrics-merge.csv
│   │   │   ├── combined_statistics.csv
│   │   │   └── cleaned_cons_combined.fasta
│   │   ├── logs/
│   │   │   ├── mge/
│   │   │   │   ├── alignment_files.log
│   │   │   │   ├── compile_barcoding_stats.log
│   │   │   │   └── {sample}_r_{r}_s_{s}/
│   │   │   ├── fasta_cleaner/fasta_cleaner_complete.txt
│   │   │   ├── rename_consensus/rename_fasta.log
│   │   │   ├── fasta_cleaner_complete.txt
│   │   │   └── exonerate_int_cleanup_complete.txt
│   │   ├── out/
│   │   ├── err/
│   │   └── {run_name}_merge-stats.csv
│   ├── concat_mode/                                           # PE mode only
│   │   └── ...                                                # Identical structure to merge_mode,
│   │                                                          # with '-concat' in place of '-merge'
│   ├── se_mode/                                               # SE/Ultima mode only
│   │   └── ...                                                # Identical structure to merge_mode,
│   │                                                          # with '-se' in place of '-merge'
│   ├── barcode_consensus_count.tsv                            # Per-sample consensus counts
│   ├── {run_name}_barcode_recovery_metrics.csv                # Combined stats (PE = both modes)
│   └── {run_name}_all_cons_combined.fasta                     # All consensus sequences
│
├── 04_barcode_validation/
│   ├── structural/                                            # Only if run_structural_validation = true
│   │   ├── structural_validation.csv
│   │   └── output_barcode_all_passing.fasta                   # All structurally passing barcodes
│   ├── taxonomic/                                             # Only if run_taxonomic_validation = true
│   │   ├── metrics/
│   │   │   ├── 01_local_blast_output.csv
│   │   │   └── 02_taxonomic_validation.csv
│   │   └── validated_barcodes.fasta
│   └── logs/
│       ├── structural_validation.log
│       ├── 01_local_blast.log
│       └── 02_taxonomic_validation.log
│
├── 05_barcoding_outcome/
│   ├── barcoding_outcome.tsv
│   ├── plots/                                                 # PNGs for MultiQC
│   │   └── plots_complete.txt
│   ├── multiqc_report/
│   │   ├── mqc_in_data/
│   │   └── multiqc_report.html
│   └── logs/
│       ├── multiqc_plots.log
│       └── multiqc.log
│
├── {run_name}_validated_barcodes.fasta                        # Only if both validations run
├── {run_name}_final_metrics.csv                               # Unified per-sample metrics
├── multiqc_report.html                                        # Convenience copy
└── logs/
```
 
</details>

 
---


# Validation process #

The BeeGees pipeline includes a barcode validation process (see [Workflow](#Workflow) section) to ensure output barcode quality is maximised through sequential structural and taxonomic validation steps, selecting the best barcode consensus sequences for downstream analyses.

Reference data required per marker:
- **COI-5P** - the [BOLDistilled](https://boldsystems.org/data/boldistilled/) BLASTn database and its taxonomy mapping (`*_SEQUENCES.fasta` and `*_TAXONOMY.tsv`, via the ['Download Source Data'](https://us-sea-1.linodeobjects.com/boldistilled/source.zip) button), plus [`COI-5p.hmm`](https://github.com/bge-barcoding/BeeGees/blob/main/resources/hmm/README.md).
- **rbcL** - the [custom reference](https://doi.org/10.6084/m9.figshare.17040680.v5) BLASTn database and taxonomy mapping (`*_dereplicated_*.fasta` and `*_dereplicated_*.tsv`, [download](https://figshare.com/ndownloader/files/56104238)), plus [`rbcL.hmm`](https://github.com/bge-barcoding/BeeGees/blob/main/resources/hmm/README.md).

> Guidance on building a custom BLASTn database and matching taxonomy TSV is in [`docs/README_custom_blast_dbs.md`](https://github.com/bge-barcoding/BeeGees/blob/main/docs/README_custom_blast_dbs.md).

>  Information on how the supported barcode HMMs were constructed can be found in [`docs/README_hmm_info.md`](https://github.com/bge-barcoding/BeeGees/blob/main/docs/README_hmm_info.md).

## Structural validation ##
Structural validation (via `structural_validation.py`) assesses every barcode consensus sequence to identify high-quality, protein-coding those suitable for taxonomic assignment. It outputs a CSV of structural, translation and quality-rank metrics for all sequences, plus `output_barcode_all_passing.fasta` containing every sequence that passes all five criteria below (a single process ID may have multiple passing sequences).

1. **Barcode region extraction** - strip tilde characters (`~`) representing missing gene regions, replace gaps (`-`) with `N`, align against the marker-specific HMM profile with nhmmer, reconstruct the barcode in HMM coordinate space, and trim leading and trailing `N`s while preserving internal ambiguous bases.
2. **Structural analysis** - calculate sequence length, gap distribution (leading/trailing/internal) and `N` count, distinguishing *original* `N`s (`barcode_ambiguous_bases_original`, indicating genuine quality issues) from processing-introduced `N`s (`barcode_ambiguous_bases`, all `N`s in the final sequence).
3. **Translation analysis** - translate in all three reading frames (0, 1, 2) using the specified genetic code, count stop codons in each, and select the frame with the fewest.
4. **Quality ranking** - assign a rank from 1 to 6 based on original `N`s, stop codons, reading frame validity and base count (lower is better):
   | Rank | Criteria |
   | --- | --- |
   | 1 | Perfect - no original `N`s, no stop codons, valid frame, ≥500 bp |
   | 2 | High quality - as above, 400–499 bp |
   | 3 | Good quality - as above, 300–399 bp |
   | 4 | Acceptable - as above, 200–299 bp |
   | 5 | Minimal - as above, 1–199 bp |
   | 6 | Problematic - contains original `N`s or translation issues |
5. **Sequence selection** - to pass structural validation and proceed to taxonomic validation, a sequence must satisfy *all* of:
   - No original `N`s (`barcode_ambiguous_bases_original == 0`)
   - No stop codons (`stop_codons == 0`)
   - A valid reading frame (`reading_frame >= 0`)
   - Sufficient informative content (`barcode_base_count > 300` bp)
   - Acceptable post-processing quality (`barcode_ambiguous_bases` < 30% of `barcode_base_count`)

## Taxonomic validation ##
Taxonomic validation runs in two steps, via `tv_local_blast.py` and `tv_blast2taxonomy.py`.

**1. Local BLASTn search.** BLASTn searches against a local database, either built from a multi-FASTA with `makeblastdb` or supplied pre-built. The e-value threshold is hardcoded to 1e-5 and `max_target_seqs` to 100. Queries are batched into chunks of 50 sequences per `blastn` invocation, with up to `threads` chunks running concurrently, so the database is loaded once per chunk rather than once per sequence. Results are demultiplexed back into one TSV per sequence (outfmt 6), each holding up to 100 hits ordered by descending percent identity, and all 100 are carried into the summary CSV. Sequences whose TSV already exists are skipped, so an interrupted run resumes without repeating finished work.
 
**2. Taxonomic assignment validation.** BLASTn results are checked against expected taxonomy using hierarchical matching and quality-based filtering:
1. Parse the local BLASTn summary CSV, per-sample expected lineages, database taxonomy mappings, and structurally validated sequences.
2. Discard hits below `min_pident` or below `min_length` (set in `beegees/config/config.yaml`).
3. Compare remaining hits against the expected lineage by exact string matching at family, genus or species level (highest rank considered is set by `taxval_rank`). The first (top) hit matching at any allowed rank is accepted.
4. Select the best sequence per process ID from those with taxonomy matches, prioritising in order: lowest matched rank (species > genus > family), then fewest gaps, fewest mismatches, highest percent identity, lowest e-value, highest alignment length, highest MGE `s` value, highest MGE `r` value, and finally sequences containing `fcleaner` in the seq ID (preferring cleaned consensus sequences).
5. Write the taxonomic validation CSV.

---

# Major outputs #

## Validated barcode FASTA ##
The primary deliverable is a multi-FASTA of all barcodes that passed both structural and taxonomic validation:

```
{output_dir}/{run_name}_validated_barcodes.fasta
```

One sequence per sample (the best-ranking barcode per process ID), with a header encoding the sample ID, MGE parameters, validation rank, and taxonomic match. Only produced when both validation steps complete successfully.

## Final metrics CSV ##
`val_csv_merger.py` merges validation outputs with preprocessing and recovery statistics into a single per-barcode-candidate CSV:

```
{output_dir}/{run_name}_final_metrics.csv
```

Consolidates:
- Read QC metrics (fastp, plus TrimGalore for PE)
- Reference retrieval results (Gene Fetch)
- Barcode recovery statistics (MitoGeneExtractor, fasta_cleaner)
- Structural validation metrics
- Taxonomic validation results

## Diagnostic plots ##
A directory of diagnostic and results PNG plots is written alongside the MultiQC report:

```
{output_dir}/05_barcoding_outcome/plots/
```

Plots cover read count distributions, alignment rates, barcode rank and mode-type breakdown, taxonomic success rates by order and phylum, percent identity histograms, and barcoding outcome summaries. These PNGs are also embedded in the MultiQC report.

## MultiQC report ##
Every run produces a self-contained interactive HTML report at:

```
{output_dir}/05_barcoding_outcome/multiqc_report/multiqc_report.html
```

A convenience copy is placed directly at `{output_dir}/multiqc_report.html`. The report integrates:
- Read QC summary (fastp)
- Barcode recovery rates (MitoGeneExtractor)
- Structural and taxonomic validation outcomes
- Per-sample barcoding success (PASS/PARTIAL/FAIL)

[View an example MultiQC report (10 samples)](docs/examples/multiqc_report.html)


---

# Screening negative controls #
If you have negative controls, such as well H12 of a 96-well plate, we recommend including them in `samples.csv` using the lowest common ancestor (LCA) of the plate as the expected taxonomy.
 
In our testing, this is often sufficient to identify inter-well contamination within a plate, whether or not a barcode consensus sequence is successfully constructed. For example, in a plate of Hymenoptera containing both Ichneumonidae and Braconidae samples, the LCA would be Hymenoptera.

---

# Citations and contributions #
 
## Cite BeeGees ##
> Parsons, D. A. J., R. A. Vos, and B. W. Price. 2026. "BeeGees: A High-Throughput Protein-Coding DNA Barcode Recovery Pipeline Tailored for Genome Skims of Museum Specimens." *Molecular Ecology Resources* 26, no. 5: e70170. https://doi.org/10.1111/1755-0998.70170
 
## Cite MitoGeneExtractor ##
BeeGees uses [MitoGeneExtractor](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.14075) at its core, so please also cite:
 
> Brasseur, M. V., Astrin, J. J., Geiger, M. F., Mayer, C. (2023). MitoGeneExtractor: Efficient extraction of mitochondrial genes from next-generation sequencing libraries. *Methods in Ecology and Evolution*. https://doi.org/10.1111/2041-210X.14075
 
## Contributing ##
Issues, forks and pull requests are welcome. This pipeline was produced by Dan Parsons @ NHMUK for the Biodiversity Genomics Europe (BGE) consortium.
 
---
 
# Future developments #
- Expand supported markers beyond COI-5P and rbcL. Will require marker-specific HMMs, BLAST databases and associated taxonomy files for barcode validation. Next likely marker to be added = MatK.
- Update 01_human_cox1_filter.py so it does not solely filter aligned reads against human COI, but instead against the whole human mitogenome.
- Replacement of final BLASTn taxonomic validation and parsing, using the [BOLDistilled](https://www.boldsystems.org/data/boldistilled/) COI database, with a SINTAX approach instead. SINTAX should be faster, does taxonomic traversal, and provides a confidence score for each match.
