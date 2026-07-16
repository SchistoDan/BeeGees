# Barcode gene Extraction and Evaluation from Genome Skims (BeeGees) Snakemake workflow #
Snakemake workflow for recovering high-quality barcode sequences at scale, built around MitoGeneExtractor and adapted for genome skims of museum specimens.

[![Snakemake](https://img.shields.io/badge/snakemake-9.9.0-brightgreen.svg?style=flat)](https://snakemake.readthedocs.io)
[![Publication DOI](https://img.shields.io/badge/DOI-10.1111%2F1755--0998.70170-blue.svg)](https://doi.org/10.1111/1755-0998.70170)

# Contents #
 - [Requirements](#Requirements)
 - [Workflow](#Workflow)
 - [Cluster configuration](#Cluster-configuration-using-Snakemake-profiles)
 - [Results structure](#Results-structure)
 - [Validation process](#Validation-process)
 - [Contributing](#Contributing)
 - [Future developments](#Future-developments)

---

# Requirements #
- [**MitoGeneExtractor**](https://github.com/cmayer/MitoGeneExtractor) version 1.9.6 (installed automatically via the conda env, see BeeGees_env.yaml).
- **Paired-end (PE) reads** in .fastq.gz or .fastq format, _or_ **single-end (SE) reads** from Ultima Genomics sequencing in .fastq.gz or .fastq format.
- **samples.csv** (generated manually, or as outlined below if working from BOLD sample metadata).
- **sequence_references.csv** (generated manually, or using [Gene Fetch](https://github.com/bge-barcoding/gene_fetch?tab=readme-ov-file) within the workflow).
- Activated **conda env** (see BeeGees_env.yaml).

---

# Workflow #
<img width="1829" height="853" alt="image" src="https://github.com/user-attachments/assets/017062f4-51c1-43b3-b9fd-6277820b45e4" />



1. **Preprocessing modes** — the pipeline automatically detects run mode from the samples.csv (see [Run mode detection](#Run-mode-detection) below):
   - **PE concat** _(paired-end)_: Adapter trimming, quality filtering, poly-G trimming, deduplication of paired-end reads using [fastp](https://github.com/OpenGene/fastp), followed by concatenation of R1+R2 reads, a secondary quality trimming with [TrimGalore](https://github.com/FelixKrueger/TrimGalore), and optional read downsampling.
   - **PE merge** _(paired-end)_: Quality control and merging of overlapping paired-end reads using [fastp](https://github.com/OpenGene/fastp), with header cleaning for MitoGeneExtractor compatibility, and optional read downsampling.
   - **SE** _(single-end / Ultima Genomics)_: Adapter trimming, poly-X tail trimming, and deduplication of single-end reads using [fastp](https://github.com/OpenGene/fastp) with Ultima-specific settings. Only one preprocessing path runs (no merge or concat steps).
2. **Sample-specific reference retrieval**: Automated retrieval of taxonomically-appropriate protein reference sequences from GenBank using [Gene-Fetch](https://github.com/bge-barcoding/gene_fetch).
3. **Barcode recovery**: Protein reference-guided extraction of barcode sequences from preprocessed reads using [MitoGeneExtractor](https://github.com/cmayer/MitoGeneExtractor), producing initial consensus sequences.
4. **Consensus sequence preparation**: Header standardisation and concatenation of raw consensus sequences into multi-FASTA format for downstream processing.
5. **Consensus cleaning and filtering pipeline (fasta_cleaner)**: Sequential quality filters applied to MGE read alignments to remove contaminants and outliers before generating cleaned consensus sequences:
   - Human COI contamination removal (common in museum specimens) ([01_human_cox1_filter.py](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/01_human_cox1_filter.py))
   - AT content filtering (removes suspected fungal/bacterial contamination) ([02_at_content_filter.py](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/02_at_content_filter.py))
   - Statistical outlier removal (eliminates reads dissimilar to initial consensus) ([03_statistical_outliers.py](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/03_statistical_outlier_filter.py))
   - Optional: Custom reference-based filtering ([04_reference_filter.py](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/04_reference_filter.py))
   - Cleaned consensus generation and metrics aggregation ([05_consensus_generator.py](https://github.com/bge-barcoding/MitoGeneExtractor-BGE/blob/main/workflow/scripts/05_consensus_generator.py))
6. **Barcode validation and selection** (see Validation Process section for more detail):
   - **Structural validation**: HMM-based barcode extraction, reading frame analysis, stop codon detection, and quality ranking of all generated barcode consensus sequences ([structural_validation.py](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/structural_validation.py))
   - **Local BLASTn search**: Parallel BLASTn searches of structurally validated barcodes against local reference database ([tv_local_blast.py](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/tv_local_blast.py))
   - **Taxonomic validation**: Hierarchical matching of BLAST results against expected taxonomy, selecting the best sequence per sample based on taxonomic match quality and alignment metrics ([tv_blast2taxonomy.py](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/tv_blast2taxonomy.py))
7. **Statistics compilation**: Aggregation of QC, recovery, cleaning, filtering, and validation metrics into comprehensive CSV reports ([compile_barcoding_stats.py](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/compile_barcoding_stats.py)).
8. **Final integration**: Merging of all pipeline metrics (read QC, MGE, fasta_cleaner, structural validation, taxonomic validation) into a unified output CSV ([val_csv_merger.py](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/val_csv_merger.py)).
9. **Evaluate barcoding outcome**: Take unified CSV file and determine barcoding success (PASS/PARTIAL/FAIL) for each sample ([barcoding_outcome.py](https://github.com/SchistoDan/BeeGees/blob/main/workflow/scripts/barcoding_outcome.py)).
10. **Cleanup**: Removal of temporary files and redundant sample-specific logs.


---


## Generate input sample CSV file ##
- This can be created manually, or via the [sample-processing](https://github.com/bge-barcoding/sample-processing) workflow.
- The file must contain the following headers: **'ID', 'forward', and 'taxid' _OR_ 'phylum->species'**. The `reverse` column is not required if running SE Ultima data.
  - `ID`: Unique sample identifier. Due to regex matching and statistics aggregation, the sample ID will be considered as the string before the first underscore. **It is therefore recommended that sample names do not use '_' characters.** E.g. BSNHM002-24 instead of BSNHM002_24, or P3-1-A10-2-G1 instead of P3_1_A10_2_G1.
  - `forward`: Absolute path to the forward (R1) read file. In fastq/fq format, either gzipped or not. For SE/Ultima runs, this is the only read file required.
  - `reverse` _(optional)_: Absolute path to the reverse (R2) read file. If provided for all samples, the run is treated as paired-end. If absent or empty for all samples, the run is treated as single-end (Ultima). **Must be consistent across all samples**. Mixed SE/PE samples in the same CSV will cause the pipeline to exit with an error.
  - `taxid` _OR_ `hierarchical taxonomy`: Unique taxonomic identifier or taxonomic lineage for sample. Taxids can be found manually by searching the expected species/genus/family of each sample in the [NCBI taxonomy database](https://www.ncbi.nlm.nih.gov/taxonomy). Alternatively, you can provide the taxonomic lineages of each sample (with the headers phylum, class, order, family, genus, species) and the corresponding taxid of the lowest identified taxonomic rank will be retrieved.
 
**samples.csv example — paired-end (taxid)**
| ID | forward | reverse | taxid |
| --- | --- | --- | --- |
| BSNHM002-24 | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | 177658 |
| BSNHM038-24 | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | 177627 |
| BSNHM046-24 | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | 3084599 |
 
**samples.csv example — single-end / Ultima (taxid)**
| ID | forward | reverse | taxid |
| --- | --- | --- | --- |
| UK001-A01 | abs/path/to/reads.fq.gz | | 177658 |
| UK001-A02 | abs/path/to/reads.fq.gz | | 177627 |
| UK001-A03 | abs/path/to/reads.fq.gz | | 3084599 |
 
**samples.csv example — paired-end (hierarchical taxonomy)**
| ID | forward | reverse | phylum | class | order | family | genus | species |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BSNHM002-24 | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | Arthropoda | Insecta | Hemiptera | Cicadidae | Tibicina | Tibicina tomentosa |
| BSNHM038-24 | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | Tracheophyta | Pinopsida | Pinales | Pinaceae | Abies | |
| BSNHM046-24 | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | Annelida | Polychaeta | Terebellida | Ampharetidae | Samytha | Samytha sexcirrata |
 
---
 

**SE/Ultima mode**
BeeGees supports single-end (SE) reads generated by Ultima Genomics (UG100) sequencing as a third preprocessing mode, alongside from the two PE ('concat' and 'merge') modes for analysis of Illumina/Element Biosciences data.
 
**Key differences from PE mode:**
- Only one preprocessing path - fastp is run in SE mode with Ultima-specific settings: poly-X tail trimming (targeting poly-T 3' artefacts characteristic of Ultima flow-based chemistry), poly-G trimming disabled (Ultima is not a two-colour system), and deduplication enabled.
- Adapter auto-detection is used as the primary adapter trimming mechanism (Ultima-specific adapter sequences **should** be trimmed by Ultima Genomic's Trimmer tool).
**Barcode recovery and downstream processing** for SE mode use the same MGE, fasta_cleaner, structural validation, and taxonomic validation steps as PE mode, with outputs written to a `se_mode/` directory (analogous to `merge_mode/` and `concat_mode/` for PE).
 
---

## Gathering sample-specific pseudo-references ##
- The sample_references.csv file can be created manually, or using [Gene-fetch](https://github.com/bge-barcoding/gene_fetch) integrated into the workflow (highly recommended). If enabled in the config.yaml by setting `run_gene_fetch` to 'true', Gene-fetch will retrieve the necessary protein pseudo-references for each sample from NCBI GenBank using the sample's taxonomic identifier (taxid) or taxonomic hierarchy. A sequence target (e.g. COI) must be specified in the config.yaml, as well as your NCBI API credentials (email address & API key - see [guidance](https://support.nlm.nih.gov/kbArticle/?pn=KA-05317) on getting a key).
- The file must contain the following header: **'ID', 'reference'name' and 'protein_reference_path'.**
  - `ID`: Unique sample identifier. This **must** be the same string as the 'ID' column in the input samples.csv file.
  - `protein_reference_path`: Absolute path to the protein pseudo-reference sequence used for sample-specific protein-guided read alignment.

**sample_references.csv example**
| ID | protein_reference_path | 
| --- | --- |
| BSNHM002-24 | path/to/BSNHM002-24.fasta |
| BSNHM038-24 | path/to/BSNHM038-24.fasta |
| BSNHM046-24 | path/toBSNHM046-24.fasta |
* **Currently, it is crucial that the sample ID (ID), the reference sequence FASTA filename, and corresponding reference sequence FASTA header are all identical for correct sample-reference file mapping.** Gene-fetch will handle this for you.

---

## Customising snakemake configuration file ##
- Update [config/config.yaml](https://github.com/bge-barcoding/BeeGees/blob/main/config/config.yaml) with the neccessary paths and variables.
```
## General BeeGees pipeline parameters and paths
run_name: BeeGees run identifier
samples_file: Path to samples.csv (see above for formatting)
sequence_reference_file: Path to sequence_references.csv (leave path blank/empty if 'run_gene_fetch' == true) (see above for formatting)
output_dir: Path to output directory. If any directories in the path do not already exist, then they wil be created

## Pre-proecssing (QC) parameters
adapter_r1: ""          # PE R1 adapter sequence to be trimmed by fastp (empty = Illumina TruSeq default)
adapter_r2: ""          # PE R2 adapter sequence to be trimmed by fastp (empty = Illumina TruSeq default)
extra_fastp_args: ""    # Accepts any additional fastp flags as single quoted strings.

## Gene Fetch parameters (https://github.com/bge-barcoding/gene_fetch)
run_gene_fetch: Set to true to use gene-fetch to generate reference sequences (default: true)
email: Email for NCBI API. Required if run_gene_fetch == true. 
api_key: NCBI API key. Required if run_gene_fetch == true.
gene: Target gene (cox1 or rbcl)
minimum_length: Minimum length (in amino acids) of protein pseudo-reference(s) to fetch (default: 500)
input_type: Taxonomic identification column(s) (taxid/hierarchical). i.e. Does the 'samples.csv' contain a 'taxid' column or 'hierarchical' taxonomic information columns (default: taxid)? (see above for formatting)
genbank: Download complete GenBank records of retrieved protein pseudo-references

## Downsampling parameters
enabled: Set to true to enable downsampling (default: false)
max_reads: Maximum number of read PAIRS to process for PE mode (e.g. 25M read pairs = 50M reads in total). For SE mode, this is the maximum number of reads directly. For either mode, setting this to zero is equivalent to 'enabled: false'

## MitoGeneExtractor parameters (https://github.com/cmayer/MitoGeneExtractor/tree/main?tab=readme-ov-file#command-line-options)
r: Exonerate relative score threshold parameter
s: Exonerate minimum score threshold parameter
n: Number of base pairs to extend beyond the Exonerate alignment
C: Genetic code to use for Exonerate (https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi)
t: Consensus threshold (e.g. 0.5 = 50%)

## Post-processing of aligned reads for cleaning and filtering (using fasta_cleaner modules)
# (01) human coi filtering -> (02) at content filtering -> (03) statistical outlier filtering -> (04, optional) reference-based filtering -> (05) 'cleaned' consensus generation -> (06) fasta_cleaner metrics aggregation
consensus_threshold: Threshold at which bases at each positional 'column' must 'agree' to be included in the consensus (e.g. 0.5 = ≥50% of bases at each position must agree)
human_threshold: Threshold at which reads are removed due to similarity with the human COI reference sequence (e.g. 0.95 = reads with ≥95% similarity are removed)
at_difference: Threshold at which reads are removed due to AT content variation (e.g. 0.1 = reads with AT% differing by 10% from the consensus are removed)
at_mode: AT content filtering mode (Absolute/Higher/Lower). Absolute = Remove sequences if AT content differs from consensus by more than threshold in either direction. Higher = Only remove sequences with AT content above at_difference threshold. Lower = Only remove sequences with AT content below at_difference threshold
outlier_percentile: Threshold at which reads are flagged as statistical outliers compared to the consensus and removed (e.g. 90.0 = reads < 90% 'similar' to the consensus are removed)
reference_dir: Path to directory containing at least one [ID].fasta file with known contaminant or target species genome(s) to be filtered or retained (see reference_filter_mode below)
reference_filter_mode: Either keep sequences that map to the supplied reference sequence (reference-based retention) or remove sequences that map to the supplied reference sequence (contaminant removal) ("keep_similar"/"remove_similar")

## Structural validation
run_structural_validation: Set false to skip structural validation step (default: true)
target: Barcode marker to extract. Corresponds to HMM files in 'resources/hmm' (cox1/coi or rbcl)
verbose: Enable verbose logging (default: false)
genetic_code: Genetic code for translation table (must be the same as 'C' MitoGeneExtractor parameter)

## Taxonomic validation
run_taxonomic_validation: Set false to skip taxonomic validation step (default: true). If run_structural_validation == false, run_taxonomic_validation MUST also == false
database: Path to directory containing BLASTn database, or to a specific FASTA file to make a BLASTn database from (using makeblastdb)
database_taxonomy: Path to TSV file containing taxonomic mappings corresponding to records in the BLASTn 'database'
taxval_rank: Taxonomic rank to stop validating at (e.g. family, genus, species) (default & recommended: family)
expected_taxonomy: Expected taxonomy file must contain the following columns: Process ID,phylum,class,order,family,genus,species. Process ID must equal 'ID' from the samples_file above. If hierarchical taxonomy information was provided in the 'samples.csv' file, this file can be reused as the expected_taxonomy CSV file required for taxonomic validation of barcode consensus sequences.
verbose: Enable verbose logging (default: true)
min_pident: Minimum percent identity (pident) threshold to be considered for returned BLAST hits. Any hit with a pident below this value is removed
min_length: Minimum length of alignment to be considered for returned BLAST hits. Any hit with a length below this value is removed

## Resource allocation for each rule
rules: Each of the main rules in the config.yaml can specify the number of requested threads and memory resources (in Mb) for every job. Rules have dynamic memory scaling upon retry (mem_mb * retry #). Make sure to change 'PARTITION' for MitoGeneExtractor, structural_validation, and taxonomic_validation rules.
```
**Currently, BeeGees-integrated barcode validation only works for COI-5P and rbcL barcodes due to HMM and BLAST database availability (to be expanded with future updates).**

---

## Cluster configuration using Snakemake profiles ##
- See `profiles/` directory for config.yaml files for 'SLURM' or 'local' cluster submission parameters. Other than the default `slurm_partition` and `jobs` parameters, all other parameters can likely stay as they are unless you experience issues.
  - The default `slurm_partition` determines the SLURM cluster partition for each snakemake job, unless otherwise specified (in the config/config.yaml). It is recommended to set this to a partition with at least 12-24 hour time limits.
  - The `jobs` parameter dictates the maximium number of workflow jobs that can be run concurrently. The value to set jobs to depends on your specific cluster. If the value is too low, it will create a bottleneck and reduce run speed/efficiency. If the value is too high, you may hit filesystem limits, job submission limits, user resource quotas, and fairshare policies, resulting in many pending or idle jobs. For example, if your cluster had a per-user memory limit of 256G, setting jobs to 20 and allocating 32G memory to each MitoGeneExtractor job would result in only 8 MitoGeneExtractor jobs running in parallel and the remaining 12 jobs to be pending until memroy is available.
- The profile (`profiles/local` or `profiles/slurm`) will need to be changed in `snakemake_run.sh` depending on your system and which one you use (see `$PROFILE` variable).

### Cluster submission ###
- Depending on your system and whether you are using the 'SLURM' or 'local' snakemake profile, there are two ways to run the BeeGees pipeline:
  - **SLURM**: Use [snakemake_run-sbatch.sh](https://github.com/SchistoDan/BeeGees/blob/main/snakemake_run-sbatch.sh). Run `sbatch snakemake_run-sbatch.sh` on the head/login node of your cluster. Submits the main snakemake coordinating job to the SLURM cluster using SBATCH, and will 'farm out' each job in the workflow to a new SBATCH job for increased parallelisation. Please change `--partition` in the SBATCH header section of the script to an appropriate cluster parition. The main snakemake coordinating job needs to run throughout the entire BeeGees run. It is therefore recommended to set this to a partition with at least 1 day-1 week time limits.
  - **local**: Use [snakemake_run.sh](https://github.com/bge-barcoding/MitoGeneExtractor-BGE/blob/main/snakemake_run.sh). Simply run `./snakemake_run.sh` on your desired cluster compute node. This node will handle all job scheduling and job computation.

---

# Results structure #
```
output_dir/
├── **01_preprocessing/**
│   ├── merge_mode/                                            # PE mode only
│   │   ├── trimmed_data/
│   │   │   ├── {sample}_merged.fq                             # Merged paired-end reads
│   │   │   ├── {sample}_merged_clean.fq                       # Header-cleaned merged reads
│   │   │   ├── {sample}_fastp_report.html                     # FastP HTML report
│   │   │   ├── {sample}_fastp_report.json                     # FastP JSON report
│   │   │   └── unpaired/                                      # Unpaired reads from merging
│   │   └── logs/
│   │       ├── clean_headers/
│   │       │   └── clean_headers.log                          # Aggregated header cleaning logs
│   │       ├── fastp/                                         # Individual FastP logs per sample
│   │       └── final_cleanup_complete.txt
│   ├── concat_mode/                                           # PE mode only
│   │   ├── trimmed_data/
│   │   │   └── {sample}/
│   │   │       ├── {sample}_R1_trimmed.fastq.gz               # Trimmed forward reads
│   │   │       ├── {sample}_R2_trimmed.fastq.gz               # Trimmed reverse reads
│   │   │       ├── {sample}_concat_trimmed.fq                 # Quality-trimmed concatenated reads
│   │   │       ├── {sample}_fastp_report.html                 # FastP HTML report
│   │   │       ├── {sample}_fastp_report.json                 # FastP JSON report
│   │   │       └── {sample}_concat.fastq_trimming_report.txt  # Trim Galore report
│   │   └── logs/
│   │       ├── concat/
│   │       │   └── concat_reads.log                           # Aggregated concatenation logs
│   │       ├── trim_galore/
│   │       │   └── trim_galore.log                            # Aggregated Trim Galore logs
│   │       ├── fastp/                                         # Individual FastP logs per sample
│   │       ├── gzip/                                          # Compression logs per sample
│   │       └── final_cleanup_complete.txt
│   └── se_mode/                                               # SE/Ultima mode only
│       ├── trimmed_data/
│       │   └── {sample}/
│       │       ├── {sample}_se_trimmed.fastq                  # Trimmed SE reads
│       │       ├── {sample}_fastp_report.html                 # FastP HTML report
│       │       └── {sample}_fastp_report.json                 # FastP JSON report
│       └── logs/
│           ├── fastp/                                         # Individual FastP logs per sample
│           └── final_cleanup_complete.txt
│
├── **02_references/**                                         # Only if run_gene_fetch = true
│   ├── protein/
│   │   └── {sample}.fasta                                     # Protein references for each sample
│   ├── genbank/                                               # GenBank records (if genbank: true)
│   └── sequence_references.csv                                # Reference metadata
│
├── **03_barcode_recovery/**
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
│   │   │   ├── fasta_cleaner/
│   │   │   │   └── fasta_cleaner_complete.txt
│   │   │   ├── rename_consensus/
│   │   │   │   └── rename_fasta.log
│   │   │   ├── fasta_cleaner_complete.txt
│   │   │   └── exonerate_int_cleanup_complete.txt
│   │   ├── out/
│   │   ├── err/
│   │   └── {run_name}_merge-stats.csv
│   ├── concat_mode/                                           # PE mode only
│   │   ├── alignment/
│   │   │   └── {sample}_r_{r}_s_{s}_align_{sample}.fas
│   │   ├── consensus/
│   │   │   ├── {sample}_r_{r}_s_{s}_con_{sample}.fas
│   │   │   └── {run_name}_cons_combined-concat.fasta
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
│   │   │   │   └── cleaned_cons_metrics-concat.csv
│   │   │   ├── combined_statistics.csv
│   │   │   └── cleaned_cons_combined.fasta
│   │   ├── logs/
│   │   │   ├── mge/
│   │   │   │   ├── alignment_files.log
│   │   │   │   ├── compile_barcoding_stats.log
│   │   │   │   └── {sample}_r_{r}_s_{s}/
│   │   │   ├── fasta_cleaner/
│   │   │   │   └── fasta_cleaner_complete.txt
│   │   │   ├── rename_consensus/
│   │   │   │   └── rename_fasta.log
│   │   │   ├── fasta_cleaner_complete.txt
│   │   │   └── exonerate_int_cleanup_complete.txt
│   │   ├── out/
│   │   ├── err/
│   │   └── {run_name}_concat-stats.csv
│   ├── se_mode/                                               # SE/Ultima mode only
│   │   ├── alignment/
│   │   │   └── {sample}_r_{r}_s_{s}_align_{sample}.fas
│   │   ├── consensus/
│   │   │   ├── {sample}_r_{r}_s_{s}_con_{sample}.fas
│   │   │   └── {run_name}_cons_combined-se.fasta
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
│   │   │   │   └── cleaned_cons_metrics-se.csv
│   │   │   ├── combined_statistics.csv
│   │   │   └── cleaned_cons_combined.fasta
│   │   ├── logs/
│   │   │   ├── mge/
│   │   │   │   ├── alignment_files.log
│   │   │   │   ├── compile_barcoding_stats.log
│   │   │   │   └── {sample}_r_{r}_s_{s}/
│   │   │   ├── fasta_cleaner/
│   │   │   │   └── fasta_cleaner_complete.txt
│   │   │   ├── rename_consensus/
│   │   │   │   └── rename_fasta.log
│   │   │   ├── fasta_cleaner_complete.txt
│   │   │   └── exonerate_int_cleanup_complete.txt
│   │   ├── out/
│   │   ├── err/
│   │   └── {run_name}_se-stats.csv
│   ├── {run_name}_barcode_recovery_metrics.csv                # Combined statistics (PE = both modes; SE = se_mode only)
│   └── {run_name}_all_cons_combined.fasta                     # All consensus sequences
│
├── **04_barcode_validation/**
│   ├── structural/                                            # Only if run_structural_validation = true
│   │   ├── structural_validation.csv
│   │   ├── {run_name}_full_sequences.fasta
│   │   └── {run_name}_barcode_sequences.fasta
│   └── taxonomic/                                             # Only if run_taxonomic_validation = true
│       ├── 01_local_blast_output.csv
│       ├── 02_taxonomic_validation.csv
│       └── {run_name}_barcode_sequences.fasta
│
├── **05_barcoding_outcome/**
│   ├── barcoding_outcome.log
│   └── barcoding_outcome.tsv
│
├── {run_name}_final_validated_barcodes.fasta                  # Only if both validations run
├── {run_name}_final_stats.csv                                 # Only if both validations run
└── logs/
```
 
---

# Validation process
The BeeGees pipeline contains an optional barcode validation process (see [Workflow](#Workflow) section and [config.yaml](https://github.com/SchistoDan/BeeGees/blob/main/config/config.yaml)) to ensure output barcode quality is maximised through sequential structural and taxonomic validation steps, selecting the best barcode consensus sequences for downstream analyses.
- The BeeGees pipeline has the capacity to validate the following barcode markers:
  - **COI-5P**: Requires the [BOLDistilled](https://boldsystems.org/data/boldistilled/) BLASTn COI database and corresponding taxonomy mapping TSV file (*_SEQUENCES.fasta & *_TAXONOMY.tsv files) downloaded via the ['Download Source Data'](https://us-sea-1.linodeobjects.com/boldistilled/source.zip) button. Utilises the [COI-5p.hmm](https://github.com/SchistoDan/BeeGees/blob/main/resources/hmm/README.md) for structural validation.
  - **rbcL**: Requires the [custom reference](https://doi.org/10.6084/m9.figshare.17040680.v5) BLASTn rbcL database and corresponding taxonomy mapping TSV file (*\_dereplicated_\*.fasta & *\_dereplicated_\*.tsv), downloaded [here](https://figshare.com/ndownloader/files/56104238). Utilises the [rbcL.hmm](https://github.com/SchistoDan/BeeGees/blob/main/resources/hmm/README.md) for structural validation.
## Structural validation
Structural validation (via `structural_validation.py`) evaluates all generated barcode consensus sequences through structural and functional analysis to identify high-quality, protein-coding sequences suitable for taxonomic assignment and species identification. Outputs a validation CSV containing comprehensive metrics for all sequences, including structural features, translation analysis, and quality ranks, and 'output_barcode_all_passing.fasta' containing ALL barcode sequences that pass the five quality criteria (multiple barcode sequences per process_id may pass).
 
**Process:**
1. Barcode region extraction: Remove tilde characters (~) representing missing gene regions, replace gap ('-') characters with ambiguous bases (N's), use nhmmer to align sequences against marker-specific HMM profiles, construct barcode sequences in HMM coordinate space, and trim leading/trailing N's while preserving internal ambiguous bases.
2. Structural analysis: Calculates sequence length, gap distribution (leading/trailing/internal), N base count, and distinguishes 'original' N's (barcode_ambiguous_bases_original, representing quality issues) from processing-introduced N's (barcode_ambiguous_bases, representing all N's in final sequence).
3. Translation analysis: Evaluates all three reading frames (0, 1, 2), translates sequences using specified genetic code, counts stop codons in each frame, and selects the optimal frame with the fewest stop codons.
4. Quality ranking: Assigns barcode ranks (1–6) based on original N's, stop codons, reading frame validity, and base count (lower = better):
     - Rank 1: Perfect sequences (no original N's, no stop codons, valid frame, ≥500bp)
     - Rank 2: High quality (no original N's, no stop codons, valid frame, 400–499bp)
     - Rank 3: Good quality (no original N's, no stop codons, valid frame, 300–399bp)
     - Rank 4: Acceptable (no original N's, no stop codons, valid frame, 200–299bp)
     - Rank 5: Minimal (no original N's, no stop codons, valid frame, 1–199bp)
     - Rank 6: Problematic (contains original N's or translation issues)
5. Sequence selection: To be considered structurally validated and proceed to taxonomic validation, sequences must pass ALL of the following criteria:
     - No original N's (barcode_ambiguous_bases_original == 0)
     - No stop codons (stop_codons == 0)
     - Sequence is in a valid reading frame (reading_frame >= 0)
     - Sufficient informative nucleotide base content (barcode_base_count > 300bp)
     - Acceptable post-processing sequence 'quality' (barcode_ambiguous_bases < 30% of barcode_base_count)
## Taxonomic validation
Taxonomic validation is a two-step process (via `tv_local_blast.py` and `tv_blast2taxonomy.py`) for verifying barcode identity through local BLAST searches and hierarchical taxonomic matching.
 
**Process:**
1. Local BLASTn search: Perform parallel BLASTn searches against a local database, either created from a multi-FASTA file (using makeblastdb), or using a pre-constructed BLAST database. The e-value threshold is hardcoded to 1e-5. In sequence-specific TSV output files (output format 6), the top 500 BLAST hits are then ordered by percent identity (in descending order). These are then filtered to the top 100 hits and are output to the summary CSV.
2. Taxonomic assignment validation: Validates BLASTn results against expected taxonomy using hierarchical matching and quality-based filtering to confirm barcode identity.
  1. Parses local BLASTn summary CSV, expected taxonomic lineage for each sample, BLAST database taxonomy mappings, and structurally validated sequences in FASTA format.
  2. Filter top 100 BLASTn hits to remove those with percent identity values below the specified threshold (< `min_pident`), as well as hits below a specified minimum alignment length (< `min_length`).
  3. Assess taxonomy of remaining BLAST hits for each sequence via hierarchical taxonomic (exact string) matching between the BLAST database taxonomy mapping and expected taxonomic lineage. Looks for matches between the expected taxonomy and database taxonomy mapping at family, genus, or species-level (highest rank to consider set with `taxval_rank`). The first (i.e. top) hit with a taxonomy match at any of the allowed ranks is accepted.
  4. Sequence selection for each sample (Process ID): Among the structurally validated consensus sequences with taxonomy matches, the 'best' sequence is selected based on the following criteria:
     - Lowest matched_rank (species > genus > family — more specific preferred)
     - Lowest gaps (alignment quality)
     - Lowest mismatches (sequence variability)
     - Highest percent identity (overall sequence similarity)
     - Lowest e-value (statistical significance)
     - Highest alignment length (matching hit confidence)
     - Highest s value (MitoGeneExtractor parameter)
     - Highest r value (MitoGeneExtractor parameter)
     - Has "fcleaner" in seq_id (prioritises cleaned consensus sequences)
  5. Generation of taxonomic validation CSV file
  
## Final metric integration
The barcode validation outputs are merged with pre-processing and barcode recovery statistics (via `val_csv_merger.py`) to create the final comprehensive BeeGees output ({run_name}_final_metrics.csv), consolidating:
- Read QC metrics (fastp, TrimGalore for PE; fastp only for SE)
- Reference retrieval results (Gene Fetch)
- Barcode recovery statistics (MGE, fasta_cleaner)
- Structural validation metrics
- Taxonomic validation results
---
 
# Citations and contributions #
 
## Cite BeeGees ##
If you use BeeGees in your work, please cite:
 
> Parsons, D. A. J., R. A. Vos, and B. W. Price. 2026. “BeeGees: A High-Throughput Protein-Coding DNA Barcode Recovery Pipeline Tailored for Genome Skims of Museum Specimens.” Molecular Ecology Resources26, no. 5: e70170. https://doi.org/10.1111/1755-0998.70170
 
## Cite MitoGeneExtractor ##
Since BeeGees uses [MitoGeneExtractor](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.14075) at its core, please also cite:
 
> Brasseur, M.V., Astrin, J.J., Geiger, M.F., Mayer, C. (2023). MitoGeneExtractor: Efficient extraction of mitochondrial genes from next-generation sequencing libraries. *Methods in Ecology and Evolution*. https://doi.org/10.1111/2041-210X.14075
 
## Contributing ##
Please feel free to submit issues, fork the repository, and create pull requests for any improvements. This pipeline was produced by Dan Parsons @ NHMUK for the Biodiversity Genomics Europe (BGE) consortium.
 
---
 
# Future developments #
- Expand supported markers beyond COI-5P and rbcL. Will require marker-specific HMMs, BLAST databases and associated taxonomy files for barcode validation. Next likely marker to be added = MatK.
- Update 01_human_cox1_filter.py so it does not solely filter aligned reads against human COI, but instead against the whole human mitogenome.
- Integrate pre-MGE contamination screening step.
- multiQC report improvements.
