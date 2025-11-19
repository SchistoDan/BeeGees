# Barcode gene Extraction and Evaluation from Genome Skims (BeeGees) Snakemake workflow #
Snakemake workflow for recovering high-quality barcode sequences at scale, built around MitoGeneExtractor and adapted for genome skims of museum specimens.


# Contents # 
 - [Requirements](#Requirements)
 - [Workflow](#Workflow)
 - [Installation and set up](#Installation-and-set-up)
 - [Cluster configuration](#Cluster-configuration-using-Snakemake-profiles)
 - [Results structure](#Results-structure)
 - [Validation process](#Validation-process)
 - [Contributing](#Contributing)
 - [Future developments](#Future-developments)


# Requirements #
- [MitoGeneExtractor](https://github.com/cmayer/MitoGeneExtractor) version 1.9.6 installed.
- Paired-end reads in .fastq.gz or .fastq format.
- samples.csv (generated manually, or as outlined below if working from BOLD sample metadata).
- sequence_references.csv (generated manually, or using [Gene Fetch](https://github.com/bge-barcoding/gene_fetch?tab=readme-ov-file) within the workflow).
- Activated conda env (see BeeGees_env.yaml).


# Workflow #
1. **Preprocessing modes** (both modes run in parallel to optimise barcode recovery from degraded hDNA):
   - **concat**: Adapter trimming, quality filtering, poly-G trimming, and deduplication of paired-end reads using [fastp](https://github.com/OpenGene/fastp), followed by concatenation of R1+R2 reads and secondary quality trimming with [TrimGalore](https://github.com/FelixKrueger/TrimGalore).
   - **merge**: Quality control and merging of overlapping paired-end reads using [fastp](https://github.com/OpenGene/fastp), with header cleaning for MitoGeneExtractor compatibility.

<p align="center">
  <img src="https://github.com/user-attachments/assets/139b8c7c-b0dc-465c-8c95-e3a58ea1ab96" width="650"/>
</p>

2. **Sample-specific reference retrieval**: Automated retrieval of taxonomically-appropriate protein reference sequences from GenBank using [Gene-Fetch](https://github.com/bge-barcoding/gene_fetch).
3. **Barcode recovery**: Protein reference-guided extraction of barcode sequences from preprocessed reads using [MitoGeneExtractor](https://github.com/cmayer/MitoGeneExtractor), producing initial consensus sequences for both preprocessing modes.
4. **Consensus sequence preparation**: Header standardisation and concatenation of raw consensus sequences into multi-FASTA format for downstream processing.
5. **Consensus cleaning and filtering pipeline (fasta_cleaner)**: Sequential quality filters applied to MGE read alignments to remove contaminants and outliers before generating cleaned consensus sequences:
   - Human COI contamination removal (common in museum specimens) ([01_human_cox1_filter.py](https://github.com/bge-barcoding/MitoGeneExtractor-BGE/blob/main/workflow/scripts/01_human_cox1_filter.py)
   - AT content filtering (removes suspected fungal/bacterial contamination) ([02_at_content_filter.py](https://github.com/bge-barcoding/MitoGeneExtractor-BGE/blob/main/workflow/scripts/02_at_content_filter.py)
   - Statistical outlier removal (eliminates reads dissimilar to initial consensus) ([03_statistical_outliers.py](https://github.com/bge-barcoding/MitoGeneExtractor-BGE/blob/main/workflow/scripts/03_statistical_outlier_filter.py)
   - Optional: Custom reference-based filtering ([04_reference_filter.py](https://github.com/bge-barcoding/MitoGeneExtractor-BGE/blob/main/workflow/scripts/04_reference_filter.py)
   - Cleaned consensus generation and metrics aggregation ([05_consensus_generator.py](https://github.com/bge-barcoding/MitoGeneExtractor-BGE/blob/main/workflow/scripts/05_consensus_generator.py)

<p align="center">
  <img width="285" height="443" alt="image" src="https://github.com/user-attachments/assets/957d43a7-0c00-40ce-bab8-1827d0e37e1b" />
</p>

6. **Barcode validation and selection** (see Validation Process section for more detail):
   - **Structural validation**: HMM-based barcode extraction, reading frame analysis, stop codon detection, and quality ranking of all generated barcode consensus sequences ([structural_validation.py](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/structural_validation.py))
   - **Local BLASTn search**: Parallel BLASTn searches of structurally validated barcodes against local reference database ([tv_local_blast.py](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/tv_local_blast.py))
   - **Taxonomic validation**: Hierarchical matching of BLAST results against expected taxonomy, selecting the best sequence per sample based on taxonomic match quality and alignment metrics ([tv_blast2taxonomy.py](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/tv_blast2taxonomy.py))
7. **Statistics compilation**: Aggregation of QC, recovery, cleaning, filtering, and validation metrics into comprehensive CSV reports for both preprocessing modes ([mge_stats.py](https://github.com/bge-barcoding/MitoGeneExtractor-BGE/blob/main/workflow/scripts/mge_stats.py)).
8. **Final integration**: Merging of all pipeline metrics (read QC, MGE, fasta_cleaner, structural validation, taxonomic validation) into a unified output CSV ([val_csv_merger.py](https://github.com/bge-barcoding/BeeGees/blob/main/workflow/scripts/val_csv_merger.py)).
9. **Cleanup**: Removal of temporary files and redundant sample-specific logs.


# Installation and set up: #
## Install MitoGeneextractors ##
-  Navigate to the [MitoGeneExtractor](https://github.com/cmayer/MitoGeneExtractor) repository and follow the [installation](https://github.com/cmayer/MitoGeneExtractor?tab=readme-ov-file#installation) instructions.
## Clone BeeGees github repository and set up conda environment ##
- [Install miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions).
```bash
git clone https://github.com/bge-barcoding/BeeGees.git [path/to/desired/install/location/]
cd installation/dir/BeeGees
conda env create -f BeeGees_env.yaml
git status
```
## Generate input sample CSV file ###
- This can be created manually, or via [sample-processing](https://github.com/bge-barcoding/sample-processing) workflow.
- The file must contian the following headers: **'ID', 'forward', 'reverse', and 'taxid' _OR_ 'phylum->species' (see below for more information).**
  - `ID`: Unique sample identifier. Due to regex matching and statistics aggregation, the sample ID will be considered as the string before the first underscore. **It is therefore recommended that sample names do not use '_' characters.** E.g. BSNHM002-24 instead of BSNHM002_24, or P3-1-A10-2-G1 instead of P3_1_A10_2_G1.
  - `forward` & `reverse`: Absolute paths to forward (R1) and reverse (R2) PE read files. In fastq/fq fomrat, either gzipped not.
  - `taxid` _OR_ `heirarchical taxonomy`: Unique taxonomic identifier or taxonomic lineage for sample. Taxid's can be found manually by searching the expected species/genus/family of each sample in the [NCBI taxonomy database](https://www.ncbi.nlm.nih.gov/taxonomy). Alternatively, you can provide the taxonomic lineages of each sample (with the headers phylum, class, order, family, genus, species) and the corresponding taxid of the lowest identified taxonomic rank will be retrieved.
  
**samples.csv example (taxid)**
| ID | forward | reverse | taxid |
| --- | --- | --- | --- |
| BSNHM002-24  | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | 177658 |
| BSNHM038-24 | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | 177627 |
| BSNHM046-24 | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | 3084599 |

**samples.csv example (hierarchical taxonomy)**
| ID | forward | reverse | phylum | class | order | family | genus | species |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| BSNHM002-24  | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | Arthropoda | Insecta | Hemiptera | Cicadidae | Tibicina | Tibicina tomentosa |
| BSNHM038-24 | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | Tracheophyta | Pinopsida | Pinales | Pinaceae | Abies |  |
| BSNHM046-24 | abs/path/to/R1.fq.gz | abs/path/to/R2.fq.gz | Annelida | Polychaeta | Terebellida | Ampharetidae | Samytha | Samytha sexcirrata |

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

## Customising snakemake configuration file ##
- Update [config/config.yaml](https://github.com/bge-barcoding/BeeGees/blob/main/config/config.yaml) with the neccessary paths and variables.
- **Currently, BeeGees barcode validaition only works for COI-5P and rbcL barcodes due to HMM and BLAST database availability (to be expanded with future updates).**
- Each of the main rules in the config.yaml can specify the number of requested threads and memory resources (in Mb) for every job (e.g. specifying 4 threads and 4G memory for fastp_pe_merge would allocate those resources for every 'fastp_pe_merge' job).
- If heirarchical taxonomy information was provided in the `samples.csv` file, this file can be reused as the expected_taxonomy CSV file required for taxonomic validation of barcode consensus sequences.

## Cluster configuration using Snakemake profiles ##
- See `profiles/` directory for config.yaml files for 'SLURM' or 'local' cluster submission parameters. Other than the default `slurm_partition` and `jobs` parameters, all other parameters can likely stay as they are unless you experience issues.
  - The default `slurm_partition` determines the SLURM cluster partition for each snakemake job, unless otherwise specified (in the config/config.yaml). It is recommended to set this to a partition with at least 12-24 hour time limits.
  - The `jobs` parameter dictates the maximium number of workflow jobs that can be run concurrently. The value to set jobs to depends on your specific cluster. If the value is too low, it will create a bottleneck and reduce run speed/efficiency. If the value is too high, you may hit filesystem limits, job submission limits, user resource quotas, and fairshare policies, resulting in many pending or idle jobs. For example, if your cluster had a per-user memory limit of 256G, setting jobs to 20 and allocating 32G memory to each MitoGeneExtractor job would result in only 8 MitoGeneExtractor jobs running in parallel and the remaining 12 jobs to be pending until memroy is available.
- The profile (`profiles/local` or `profiles/slurm`) will need to be changed in `snakemake_run.sh` depending on your system and which one you use (see `$PROFILE` variable).

### Cluster submission ###
- Depending on your system and whether you are using the 'SLURM' or 'local' snakemake profile, there are two ways to run the BeeGees pipeline:
  - **SLURM**: Use [snakemake_run-sbatch.sh](https://github.com/SchistoDan/BeeGees/blob/main/snakemake_run-sbatch.sh). Run `sbatch snakemake_run-sbatch.sh` on the head/login node of your cluster. Submits the main snakemake coordinating job to the SLURM cluster using SBATCH, and will 'farm out' each job in the workflow to a new SBATCH job for increased parallelisation. Please change `--partition` in the SBATCH header section of the script to an appropriate cluster parition. The main snakemake coordinating job needs to run throughout the entire BeeGees run. It is therefore recommended to set this to a partition with at least 1 day-1 week time limits.
  - **local**: Use [snakemake_run.sh](https://github.com/bge-barcoding/MitoGeneExtractor-BGE/blob/main/snakemake_run.sh). Simply run `./snakemake_run.sh` on your desired cluster compute node. This node will handle all job scheduling and job computation.



# Results structure #
```
output_dir/
├── **01_preprocessing/**
│   ├── merge_mode/
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
│   └── concat_mode/
│       ├── trimmed_data/
│       │   └── {sample}/
│       │       ├── {sample}_R1_trimmed.fastq.gz               # Trimmed forward reads
│       │       ├── {sample}_R2_trimmed.fastq.gz               # Trimmed reverse reads
│       │       ├── {sample}_concat_trimmed.fq                 # Quality-trimmed concatenated reads
│       │       ├── {sample}_fastp_report.html                 # FastP HTML report
│       │       ├── {sample}_fastp_report.json                 # FastP JSON report
│       │       └── {sample}_concat.fastq_trimming_report.txt  # Trim Galore report
│       └── logs/
│           ├── concat/
│           │   └── concat_reads.log                           # Aggregated concatenation logs
│           ├── trim_galore/
│           │   └── trim_galore.log                            # Aggregated Trim Galore logs
│           ├── fastp/                                         # Individual FastP logs per sample
│           ├── gzip/                                          # Compression logs per sample
│           └── final_cleanup_complete.txt
│
├── **02_references/**                                             # Only if run_gene_fetch = true
│   ├── protein/
│   │   └── {sample}.fasta                                     # Protein references for each sample
│   ├── genbank/                                               # GenBank records (if genbank: true)
│   └── sequence_references.csv                                # Reference metadata
│
├── **03_barcode_recovery/**
│   ├── merge_mode/
│   │   ├── alignment/
│   │   │   └── {sample}_r_{r}_s_{s}_align_{sample}.fas        # MGE alignment files
│   │   ├── consensus/
│   │   │   ├── {sample}_r_{r}_s_{s}_con_{sample}.fas          # Individual consensus files
│   │   │   └── {run_name}_cons_combined-merge.fasta           # Combined consensus sequences
│   │   ├── fasta_cleaner/
│   │   │   ├── 01_human_filtered/
│   │   │   │   ├── human_filtered.txt                         # List of filtered files
│   │   │   │   └── human_filter_metrics.csv                   # Human filtering metrics
│   │   │   ├── 02_at_filtered/
│   │   │   │   ├── at_filtered_sequences/                     # Individual filtered files
│   │   │   │   ├── at_filtered.txt                            # List of filtered files
│   │   │   │   └── at_filter_summary.csv                      # AT filtering summary
│   │   │   ├── 03_outlier_filtered/
│   │   │   │   ├── outlier_filtered.txt                       # List of filtered files
│   │   │   │   ├── outlier_filter_summary_metrics.csv         # Summary metrics
│   │   │   │   └── outlier_filter_individual_metrics.csv      # Individual metrics
│   │   │   ├── 04_reference_filtered/                         # Optional - if reference filtering enabled
│   │   │   │   ├── reference_filtered.txt                     # List of filtered files
│   │   │   │   └── reference_filter_metrics.csv               # Reference filtering metrics
│   │   │   ├── 05_cleaned_consensus/
│   │   │   │   └── cleaned_cons_metrics-merge.csv             # Consensus generation metrics
│   │   │   ├── combined_statistics.csv                        # Aggregated cleaning statistics
│   │   │   └── cleaned_cons_combined.fasta                    # Final cleaned consensus sequences
│   │   ├── logs/
│   │   │   ├── mge/
│   │   │   │   ├── alignment_files.log                        # List of alignment files
│   │   │   │   ├── mge_stats.log                              # MGE statistics log
│   │   │   │   └── {sample}_r_{r}_s_{s}/                      # MGE vulgar files per sample
│   │   │   ├── fasta_cleaner/
│   │   │   │   └── fasta_cleaner_complete.txt                 # Cleaner completion flag
│   │   │   ├── rename_consensus/
│   │   │   │   └── rename_fasta.log                           # Header renaming logs
│   │   │   ├── fasta_cleaner_complete.txt                     # Main cleaner completion
│   │   │   └── exonerate_int_cleanup_complete.txt             # Intermediate cleanup completion
│   │   ├── out/                                               # MGE output files per sample/parameter
│   │   ├── err/                                               # MGE error logs per sample/parameter
│   │   └── {run_name}_merge-stats.csv                         # Mode-specific statistics
│   └── concat_mode/
│       ├── alignment/
│       │   └── {sample}_r_{r}_s_{s}_align_{sample}.fas        # MGE alignment files
│       ├── consensus/
│       │   ├── {sample}_r_{r}_s_{s}_con_{sample}.fas          # Individual consensus files
│       │   └── {run_name}_cons_combined-concat.fasta          # Combined consensus sequences
│       ├── fasta_cleaner/
│       │   ├── 01_human_filtered/
│       │   │   ├── human_filtered.txt                         # List of filtered files
│       │   │   └── human_filter_metrics.csv                   # Human filtering metrics
│       │   ├── 02_at_filtered/
│       │   │   ├── at_filtered_sequences/                     # Individual filtered files
│       │   │   ├── at_filtered.txt                            # List of filtered files
│       │   │   └── at_filter_summary.csv                      # AT filtering summary
│       │   ├── 03_outlier_filtered/
│       │   │   ├── outlier_filtered.txt                       # List of filtered files
│       │   │   ├── outlier_filter_summary_metrics.csv         # Summary metrics
│       │   │   └── outlier_filter_individual_metrics.csv      # Individual metrics
│       │   ├── 04_reference_filtered/                         # Optional - if reference filtering enabled
│       │   │   ├── reference_filtered.txt                     # List of filtered files
│       │   │   └── reference_filter_metrics.csv               # Reference filtering metrics
│       │   ├── 05_cleaned_consensus/
│       │   │   └── cleaned_cons_metrics-concat.csv            # Consensus generation metrics
│       │   ├── combined_statistics.csv                        # Aggregated cleaning statistics
│       │   └── cleaned_cons_combined.fasta                    # Final cleaned consensus sequences
│       ├── logs/
│       │   ├── mge/
│       │   │   ├── alignment_files.log                        # List of alignment files
│       │   │   ├── mge_stats.log                              # MGE statistics log
│       │   │   └── {sample}_r_{r}_s_{s}/                      # MGE vulgar files per sample
│       │   ├── fasta_cleaner/
│       │   │   └── fasta_cleaner_complete.txt                 # Cleaner completion flag
│       │   ├── rename_consensus/
│       │   │   └── rename_fasta.log                           # Header renaming logs
│       │   ├── fasta_cleaner_complete.txt                     # Main cleaner completion
│       │   └── exonerate_int_cleanup_complete.txt             # Intermediate cleanup completion
│       ├── out/                                               # MGE output files per sample/parameter
│       ├── err/                                               # MGE error logs per sample/parameter
│       └── {run_name}_concat-stats.csv                        # Mode-specific statistics
│   ├── {run_name}_BeeGees_stats.csv                              # Combined statistics from both modes
│   └── {run_name}_all_cons_combined.fasta                     # All consensus sequences from both modes
│
├── **04_barcode_validation/**
│   ├── structural/                                            # Only if run_structural_validation = true
│   │   ├── structural_validation.csv                          # Structural validation results
│   │   ├── {run_name}_full_sequences.fasta                    # Full sequences passing validation
│   │   └── {run_name}_barcode_sequences.fasta                 # Barcode sequences passing validation
│   └── taxonomic/                                             # Only if run_taxonomic_validation = true (& run_structural_validation = true)
│       ├── 01_local_blast_output.csv                          # BLAST results
│       ├── 02_taxonomic_validation.csv                        # Taxonomic validation results
│       └── {run_name}_barcode_sequences.fasta                 # Final validated barcode sequences
│
├── {run_name}_final_validated_barcodes.fasta                  # Only if both validations run
├── {run_name}_final_stats.csv                                 # Only if both validations run
└── logs/                                                      # Top-level logs directory
```


# Validation process
The BeeGees pipeline contains an optional barcode validation process (see [Workflow](#Workflow) section and [config.yaml](https://github.com/SchistoDan/BeeGees/blob/main/config/config.yaml)) to ensure output barcode quality is maximised through sequential structural and taxonomic validation steos, selecting the best barcode consensus sequences for downstream analyses. 
- The BeeGees pipeline has the capacity to validate the following barcode markers:
  - **COI-5P**: Requires the [BOLDistilled](https://boldsystems.org/data/boldistilled/) BLASTn COI database and corresponding taxonomy mapping TSV file (*_SEQUENCES.fasta & *_TAXONOMY.tsv files) downloaded via the ['Download Source Data'](https://us-sea-1.linodeobjects.com/boldistilled/source.zip) button.
  - **rbcL**: Requires the [custom reference](https://doi.org/10.6084/m9.figshare.17040680.v5) BLASTn rbcL database and corresponding taxonomy mapping TSV file (*\_dereplicated_\*.fasta & *\_dereplicated_\*.tsv), downloaded [here](https://figshare.com/ndownloader/files/56104238).

## Structural validation
Structural validation (via `structural_validation.py`) evaluates all generated barcode consensus sequences (from both pre-processing mode and all fasta_cleaner variants) through structural and functional analysis to identify high-quality, protein-coding sequences suitable for taxonomic assignment and species identification. Outputs  a validation CSV containing comprehensive metrics for all sequences, including structural features, translation analysis, and quality ranks, and 'output_barcode_all_passing.fasta' containing ALL barcode sequences that pass the five quality criteria (multiple barcode sequences per process_id may pass)

**Process:**
1. Barcode region extraction: Remove tilde characters (~) representing missing gene regions, replace gap ('-') characters with ambiguous bases (N's), use nhmmer to align sequences against marker-specific HMM profiles, constructs barcode sequences in HMM coordinate space, and trims leading/trailing N's while preserving internal ambiguous bases.
2. Structural analysis: Calculates sequence length, gap distribution (leading/trailing/internal), N base count, and distinguishes 'original' N's (barcode_ambiguous_bases_original, representing quality issues) from processing-introduced N's (barcode_ambiguous_bases, representing all N's in final sequence).
3. Translation analysis: Evaluates all three reading frames (0, 1 2), translates sequences using specified genetic code, counts stop codons in each frame, and selects the optimal frame with the fewest stop codons.
4. Quality ranking: Assigns barcode ranks (1-6) based on original N's, stop codons, reading frame validity, and base count (lower = better):
     - Rank 1: Perfect sequences (no original N's, no stop codons, valid frame, ≥500bp)
     - Rank 2: High quality (no original N's, no stop codons, valid frame, 400-499bp)
     - Rank 3: Good quality (no original N's, no stop codons, valid frame, 300-399bp)
     - Rank 4: Acceptable (no original N's, no stop codons, valid frame, 200-299bp)
     - Rank 5: Minimal (no original N's, no stop codons, valid frame, 1-199bp)
     - Rank 6: Problematic (contains original N's or translation issues)
5. Sequence selection: To be considered structurally validated and proceed to taxonomic validation, sequences mut pass ALL of the following criteria:
     - No original N's ( barcode_ambiguous_bases_original == 0)
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
  2. Filter top 100 BLASTn hits to remove those with percent identity values below the specified threshold (< `min_pident`), as well as hits below a specified minimum aignment length (< `min_length`).
  3. Assess taxonomy of remaining BLAST hits for each sequence via hierarchical taxonomic (exact string) matching between the BLAST database taxonomy mapping and expected taxonomic lineage. Looks for matches between the expected taxonomy and database taxonomy mapping at family, genus, or species-level (highest rank to consider set with `taxval_rank`). The first (i.e. top) hit with a taxonomy match at any of the allowed ranks is accepted.
  4. Sequence selection for each sample (Process ID): Among the structurally validated consensus sequences with taxonomy matches, the 'best' sequence is selected based on the following criteria:
     - Lowest matched_rank (species > genus > family - more specific preferred)
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
The barcode validation outputs are merged with pre-processing and barcode recovery statistics (via `val_csv_merger.py`) to create the final comprehensive BeeGees output ({run_name}_final_stats.csv), consolidating:
- Read QC metrics (fastp, TrimGalore)
- Reference retrieval results (Gene Fetch)
- Barcode recovery statistics (MGE, fasta_cleaner)
- Structural validation metrics
- Taxonomic validation results



# Contributing #
- Please feel free to submit issues, fork the repository, and create pull requests for any improvements.
- This snakemake pipeline was produced by Dan Parsons @ NHMUK for the Biodiversity Genomics Europe (BGE) consortium. If you use BeeGees in your work, please cite our paper at ...
- Since BeeGees uses [MitogeneExtractor](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.14075) at its core, please cite:
  Brasseur, M.V., Astrin, J.J., Geiger, M.F., Mayer, C., 2023. MitoGeneExtractor: Efficient extraction of mitochondrial genes from next-generation sequencing libraries. Methods in Ecology and Evolution.


# Future developments #
- Expand supported markers beyond COI-5P and rbcL. Will require marker-specific HMMs, BLAST databases and associated taxonomy files for barcode validation. Next likely maker to be added = Matk.
- Increase flexibility of input sequence_references CSV headers, so that ID/id/Process ID/PROCESS ID/process_id/sample/sample_id/SAMPLE ID/etc are accepted.
- Update 01_human_cox1_filter.py so it does not solely filter aligned reads against human COI, but instead against the whole human mitogenome.
- Integrate pre-MGE contamination screening step (e.g. using BBDuk)
- Make downsampling a separate step that happens after fastp trimming for both merge_mode and concat_mode.
- Output a multi-fasta file of sequences which passed structural validation but failed taxonomy for each Process-ID.
- Output simple plots.
- Clean up and refactor final CSV output, and remove unnecessary columns.
  
