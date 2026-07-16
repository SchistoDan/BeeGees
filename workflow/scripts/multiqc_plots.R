#!/usr/bin/env Rscript

library(dplyr)
library(ggplot2)
library(yaml)
library(scales)

# ============================================================================
# TESTING - Set paths here for direct execution/testing, leave NULL for
# paths passed via commandArgs
# ============================================================================
TEST_INPUT_CSV          <- ""
TEST_INPUT_OUTCOME      <- ""
TEST_OUTPUT_DIR         <- ""
TEST_FASTP_CONCAT_CSV   <- ""
TEST_FASTP_MERGE_CSV    <- ""
TEST_FINAL_METRICS_CSV  <- ""
TEST_TAXDUMP_DIR        <- ""

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 7) {
  cat("Running with command-line arguments\n")
  input_csv           <- args[1]
  input_outcome       <- args[2]
  output_dir          <- args[3]
  fastp_concat_csv    <- args[4]
  fastp_merge_csv     <- args[5]
  final_metrics_csv   <- args[6]
  taxdump_dir         <- args[7]
  run_mode            <- if (length(args) >= 8) toupper(args[8]) else "PE"
} else if (nchar(TEST_INPUT_CSV) > 0 && nchar(TEST_OUTPUT_DIR) > 0) {
  cat("Running in TEST mode\n")
  input_csv           <- TEST_INPUT_CSV
  input_outcome       <- TEST_INPUT_OUTCOME
  output_dir          <- TEST_OUTPUT_DIR
  fastp_concat_csv    <- TEST_FASTP_CONCAT_CSV
  fastp_merge_csv     <- TEST_FASTP_MERGE_CSV
  final_metrics_csv   <- TEST_FINAL_METRICS_CSV
  taxdump_dir         <- TEST_TAXDUMP_DIR
  run_mode            <- "PE"
} else {
  stop("Usage: Rscript multiqc_plots.R <input_csv> <input_outcome_tsv> <output_dir> <fastp_concat_csv> <fastp_merge_csv> <final_metrics_csv> <taxdump_dir> [run_mode]")
}

# Run mode: "PE" (concat + merge preprocessing) or "SE" (single-end Ultima).
# In SE runs there is a single fastp summary; it is passed in the concat slot
# and the merge data frame is left empty.

if (!run_mode %in% c("PE", "SE")) run_mode <- "PE"

# Label used in per-sample QC plot descriptions (the "primary" preprocessing mode).
primary_mode_label <- if (run_mode == "SE") "se" else "concat"

cat("Input CSV:           ", input_csv, "\n")
cat("Input outcome:       ", input_outcome, "\n")
cat("Output dir:          ", output_dir, "\n")
cat("Fastp concat CSV:    ", fastp_concat_csv, "\n")
cat("Fastp merge CSV:     ", fastp_merge_csv, "\n")
cat("Final metrics CSV:   ", final_metrics_csv, "\n")
cat("Taxdump dir:         ", taxdump_dir, "\n")
cat("Run mode:            ", run_mode, "\n\n")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================================
# READ AND PROCESS: sequence_references.csv
# ============================================================================
df <- read.csv(input_csv, stringsAsFactors = FALSE)

cat("sequence_references dimensions:", nrow(df), "rows,", ncol(df), "cols\n")
cat("Columns:", paste(colnames(df), collapse = " | "), "\n\n")

df$taxonomic_rank_raw <- sub(":.*", "", df$matched_rank)

df$taxonomic_rank <- dplyr::case_when(
  df$taxonomic_rank_raw %in% c("class", "subclass", "infraclass")              ~ "class",
  df$taxonomic_rank_raw %in% c("order", "suborder", "infraorder", "parvorder") ~ "order",
  df$taxonomic_rank_raw %in% c("family", "subfamily", "superfamily")           ~ "family",
  df$taxonomic_rank_raw %in% c("genus", "subgenus")                            ~ "genus",
  df$taxonomic_rank_raw %in% c("species")                                      ~ "species",
  df$taxonomic_rank_raw %in% c("tribe")                                        ~ "tribe",
  df$taxonomic_rank_raw %in% c("clade")                                        ~ "clade",
  TRUE ~ df$taxonomic_rank_raw
)

rank_order <- c("species", "genus", "family", "tribe", "order", "class", "clade")

# ============================================================================
# READ AND PROCESS: barcoding_outcome.tsv
# ============================================================================
df_outcome <- read.delim(input_outcome, stringsAsFactors = FALSE, sep = "\t")

cat("barcoding_outcome dimensions:", nrow(df_outcome), "rows,", ncol(df_outcome), "cols\n")
cat("Columns:", paste(colnames(df_outcome), collapse = " | "), "\n\n")

df_outcome <- df_outcome %>%
  mutate(
    expected_species = trimws(sub(".*;", "", expected_taxonomy)),
    observed_species = ifelse(
      !is.na(observed_taxonomy) & observed_taxonomy != "NA",
      trimws(sub("^[^:]+:\\s*([^(]+)\\s*\\(species\\).*", "\\1", observed_taxonomy)),
      NA_character_
    ),
    outcome_category = dplyr::case_when(
      barcode_outcome == "FAIL"    ~ "Fail",
      barcode_outcome == "PARTIAL" ~ "Partial",
      barcode_outcome == "PASS"    ~ "Pass",
      TRUE                         ~ "Unknown"
    )
  )

outcome_levels <- c("Pass", "Partial", "Fail")
df_outcome$outcome_category <- factor(df_outcome$outcome_category,
                                      levels = outcome_levels)

cat("Outcome category summary:\n")
print(table(df_outcome$outcome_category, useNA = "ifany"))
cat("\n")

# ============================================================================
# READ AND PROCESS: fastp summary CSVs
# ============================================================================
# PE: arg4 = concat summary, arg5 = merge summary.
# SE: arg4 = se summary (passed in the concat slot); merge is left empty so the
#     merge-only sections degrade gracefully to zero rows.
df_fastp_concat <- read.csv(fastp_concat_csv, stringsAsFactors = FALSE)

if (run_mode == "SE") {
  # Empty data frame with the same schema as the primary (se) summary.
  df_fastp_merge <- df_fastp_concat[0, , drop = FALSE]
} else {
  df_fastp_merge <- read.csv(fastp_merge_csv, stringsAsFactors = FALSE)
}

# Force the numeric QC columns to numeric. For SE, insert_size_peak is always
# empty (paired-end-only metric), which read.csv would otherwise type as a
# logical all-NA column and break the continuous-scale density plot.
.fastp_numeric_cols <- c("duplication_rate", "insert_size_peak",
                         "before_total_reads", "after_total_reads")
for (.col in .fastp_numeric_cols) {
  if (.col %in% names(df_fastp_concat)) df_fastp_concat[[.col]] <- suppressWarnings(as.numeric(df_fastp_concat[[.col]]))
  if (.col %in% names(df_fastp_merge))  df_fastp_merge[[.col]]  <- suppressWarnings(as.numeric(df_fastp_merge[[.col]]))
}

cat("fastp primary (", primary_mode_label, ") dimensions:", nrow(df_fastp_concat), "rows,", ncol(df_fastp_concat), "cols\n")
cat("fastp merge dimensions: ", nrow(df_fastp_merge),  "rows,", ncol(df_fastp_merge),  "cols\n\n")


# ============================================================================
# PLOT 1: Taxonomic rank bar chart (MultiQC bargraph)
# ============================================================================
rank_counts <- df %>%
  group_by(taxonomic_rank) %>%
  summarise(count = n(), .groups = "drop") %>%
  mutate(taxonomic_rank = factor(taxonomic_rank, levels = rank_order)) %>%
  filter(!is.na(taxonomic_rank)) %>%
  arrange(taxonomic_rank)

cat("Taxonomic ranks found:\n")
print(rank_counts)
cat("\n")

y_max     <- max(rank_counts$count, na.rm = TRUE)
y_ceiling <- max(50, ceiling(y_max / 50) * 50)

mqc_taxrank <- list(
  id          = "Reference_taxonomic_rank_distribution",
  parent_id   = "gene_fetch",
  parent_name = "Gene Fetch",
  description = paste0(
    "Taxonomic rank of retrieved protein pseudo-reference sequences across all samples ",
    "(n = ", nrow(df), " total sequences)."
  ),
  plot_type    = "bargraph",
  pconfig      = list(
    id       = "gene_fetch_taxrank_bargraph",
    title    = "Gene Fetch: Matched Rank",
    ylab     = "Number of sequences",
    ymax     = y_ceiling,
    cpswitch = FALSE
  ),
  data = setNames(
    lapply(rank_counts$count, function(n) list("Sequences" = n)),
    as.character(rank_counts$taxonomic_rank)
  )
)

yaml::write_yaml(mqc_taxrank, file.path(output_dir, "gene_fetch_01_taxrank_mqc.yaml"))
cat("Written: gene_fetch_01_taxrank_mqc.yaml\n")

# ============================================================================
# PLOT 2: Barcoding outcome summary bargraph (MultiQC bargraph)
# ============================================================================
outcome_counts <- df_outcome %>%
  group_by(outcome_category, .drop = FALSE) %>%
  summarise(count = n(), .groups = "drop")

outcome_data <- list(
  "All samples" = setNames(
    as.list(outcome_counts$count),
    as.character(outcome_counts$outcome_category)
  )
)

y_ceiling_outcome <- max(50, ceiling(sum(outcome_counts$count) / 50) * 50)

mqc_outcome <- list(
  id                 = "barcoding_outcome_summary",
  parent_id          = "barcoding_outcome",
  parent_name        = "Barcoding Outcome",
  parent_description = paste0(
    "Summary of barcoding outcomes across all ", nrow(df_outcome), " samples.<br>",
    "<br>",
    "<b>Pass</b>: barcode successfully recovered and validated.<br>",
    "<b>Partial</b>: barcode recovered but failed structural validation.<br>",
    "<b>Fail</b>: no barcode recovered.<br>",
    "<br>",
    "For a barcode to be considered a 'Pass', it must successfully undergo structural and taxonomic validation. ",
    "Structural validation consists of ambiguous base (N), stop codon, reading frame, and base content checks. ",
    "Taxonomic validation comprises a BLASTn search against an appropriate database ",
    "(e.g. BOLDistilled for COI) and a confident match to family-level or lower."
  ),
  plot_type    = "bargraph",
  pconfig      = list(
    id       = "barcoding_outcome_bargraph",
    title    = "Barcoding Outcome Summary",
    ylab     = "Number of samples",
    ymax     = y_ceiling_outcome,
    cpswitch = FALSE
  ),
  data = outcome_data
)

yaml::write_yaml(mqc_outcome, file.path(output_dir, "barcoding_01_outcome_summary_mqc.yaml"))
cat("Written: barcoding_01_outcome_summary_mqc.yaml\n")

# ============================================================================
# PLOT 3: Per-sample barcoding outcome table (MultiQC generalstats)
# ============================================================================
generalstats_data <- setNames(
  lapply(seq_len(nrow(df_outcome)), function(i) {
    list(
      "Outcome"          = as.character(df_outcome$outcome_category[i]),
      "Expected species" = df_outcome$expected_species[i],
      "Observed species" = ifelse(is.na(df_outcome$observed_species[i]),
                                  "NA", df_outcome$observed_species[i])
    )
  }),
  df_outcome$ID
)

mqc_generalstats <- list(
  id          = "barcoding_outcome_table",
  parent_id   = "barcoding_outcome",
  parent_name = "Barcoding Outcome",
  plot_type    = "generalstats",
  pconfig      = list(
    list(
      "Outcome"          = list(title = "Outcome",
                                description = "Barcoding outcome category"),
      "Expected species" = list(title = "Expected species",
                                description = "Expected species from taxonomy file"),
      "Observed species" = list(title = "Observed species",
                                description = "Top BLAST hit species")
    )
  ),
  data = generalstats_data
)

yaml::write_yaml(mqc_generalstats, file.path(output_dir, "barcoding_02_per_sample_mqc.yaml"))
cat("Written: barcoding_02_per_sample_mqc.yaml\n")

# ============================================================================
# PLOT 4: Insert size peak distribution - concat only (MultiQC linegraph)
# ============================================================================

insert_concat <- df_fastp_concat$insert_size_peak[!is.na(df_fastp_concat$insert_size_peak)]

cat("Insert size peak summary - Concat:\n")
print(summary(insert_concat))
cat("\n")

# ============================================================================
# PLOT 5: Duplication rate - concat only (MultiQC linegraph)
# ============================================================================

build_dup_rate_bins <- function(x, binwidth = 0.01) {
  x <- x[is.finite(x)]
  # No finite data (e.g. empty or all-NA duplication column): return an empty
  # series so the report degrades gracefully instead of halting.
  if (length(x) == 0) {
    return(setNames(list(), character(0)))
  }
  lo <- floor(min(x) / binwidth) * binwidth
  hi <- ceiling(max(x) / binwidth) * binwidth
  if (hi <= lo) hi <- lo + binwidth  # guarantee >= 2 breaks when all values are equal
  breaks <- seq(lo, hi, by = binwidth)
  counts <- hist(x, breaks = breaks, plot = FALSE)
  setNames(as.list(as.integer(counts$counts)), as.character(round(counts$mids, 4)))
}

dup_concat <- df_fastp_concat$duplication_rate[!is.na(df_fastp_concat$duplication_rate)]

cat("Duplication rate summary - Concat:\n")
print(summary(dup_concat))
cat("\n")

mqc_dup_rate <- list(
  id          = "fastp_duplication_rate",
  parent_id   = "preprocessing_qc",
  parent_name = "Preprocessing QC",
  description = paste0(
    "Distribution of duplication rate across samples."),
  plot_type    = "linegraph",
  pconfig      = list(
    id     = "fastp_duplication_rate_linegraph",
    title  = "Fastp: Duplication Rate Distribution",
    xlab   = "Duplication rate",
    ylab   = "Number of samples",
    xmin   = 0
  ),
  data = setNames(
    list(build_dup_rate_bins(dup_concat)),
    if (run_mode == "SE") "SE" else "Concat"
  )
)

yaml::write_yaml(mqc_dup_rate, file.path(output_dir, "fastp_02_duplication_rate_mqc.yaml"))
cat("Written: fastp_02_duplication_rate_mqc.yaml\n")

# ============================================================================
# PLOT 5b: Per-sample fastp metrics table (MultiQC table)
# ============================================================================

fastp_table_cols <- c(
  "before_total_reads", "before_total_bases",
  "after_total_reads",  "after_total_bases",
  "after_q30_rate",     "after_gc_content",
  "duplication_rate",   "insert_size_peak"
)

fastp_table_data <- setNames(
  lapply(seq_len(nrow(df_fastp_concat)), function(i) {
    list(
      "Before reads"      = df_fastp_concat$before_total_reads[i],
      "Before bases"      = df_fastp_concat$before_total_bases[i],
      "After reads"       = df_fastp_concat$after_total_reads[i],
      "After bases"       = df_fastp_concat$after_total_bases[i],
      "Q30 rate (after)"  = df_fastp_concat$after_q30_rate[i],
      "GC content (after)"= df_fastp_concat$after_gc_content[i],
      "Duplication rate"  = df_fastp_concat$duplication_rate[i],
      "Insert size peak"  = df_fastp_concat$insert_size_peak[i]
    )
  }),
  df_fastp_concat$sample_name
)

mqc_fastp_table <- list(
  id          = "fastp_per_sample_table",
  parent_id   = "preprocessing_qc",
  parent_name = "Preprocessing QC",
  description = sprintf("Per-sample fastp QC metrics for %s mode.", primary_mode_label),
  plot_type   = "table",
  pconfig     = list(
    id      = "fastp_per_sample_table_config",
    title   = "Fastp: Per-Sample QC Metrics"
  ),
  headers = list(
    "Before reads"       = list(title = "Before reads",
                                description = "Total reads before filtering",
                                format = "{:,.0f}"),
    "Before bases"       = list(title = "Before bases",
                                description = "Total bases before filtering",
                                format = "{:,.0f}"),
    "After reads"        = list(title = "After reads",
                                description = "Total reads after filtering",
                                format = "{:,.0f}"),
    "After bases"        = list(title = "After bases",
                                description = "Total bases after filtering",
                                format = "{:,.0f}"),
    "Q30 rate (after)"   = list(title = "Q30 rate (after)",
                                description = "Fraction of bases >= Q30 after filtering",
                                min = 0, max = 1,
                                format = "{:.3f}"),
    "GC content (after)" = list(title = "GC content (after)",
                                description = "GC content after filtering",
                                min = 0, max = 1,
                                format = "{:.3f}"),
    "Duplication rate"   = list(title = "Duplication rate",
                                description = "Estimated duplication rate",
                                min = 0, max = 1,
                                format = "{:.3f}"),
    "Insert size peak"   = list(title = "Insert size peak",
                                description = "Peak of insert size distribution (bp)",
                                format = "{:,.0f}")
  ),
  data = fastp_table_data
)

yaml::write_yaml(mqc_fastp_table, file.path(output_dir, "fastp_03_per_sample_table_mqc.yaml"))
cat("Written: fastp_03_per_sample_table_mqc.yaml\n")

# ============================================================================
# PLOT 5c: Per-sample before_total_reads bargraph (MultiQC bargraph)
# ============================================================================

reads_y_max     <- max(df_fastp_concat$before_total_reads, na.rm = TRUE)
reads_y_ceiling <- max(1000000, ceiling(reads_y_max / 1000000) * 1000000)

mqc_reads_bargraph <- list(
  id          = "fastp_before_total_reads",
  parent_id   = "preprocessing_qc",
  parent_name = "Preprocessing QC",
  description = sprintf("Total reads before filtering by sample (%s mode).", primary_mode_label),
  plot_type   = "bargraph",
  pconfig     = list(
    id       = "fastp_before_total_reads_bargraph",
    title    = "Fastp: Before Total Reads per Sample",
    ylab     = "Number of reads",
    ymax     = reads_y_ceiling,
    cpswitch = FALSE
  ),
  data = setNames(
    lapply(df_fastp_concat$before_total_reads,
           function(n) list("Before total reads" = n)),
    df_fastp_concat$sample_name
  )
)

yaml::write_yaml(mqc_reads_bargraph, file.path(output_dir, "fastp_04_before_total_reads_mqc.yaml"))
cat("Written: fastp_04_before_total_reads_mqc.yaml\n")

# ============================================================================
# LOAD TAXDUMP: Extract authoritative phylum names from nodes.dmp + names.dmp
# ============================================================================
cat("Locating taxdump files in:", taxdump_dir, "\n")

nodes_path <- file.path(taxdump_dir, "nodes.dmp")
names_path <- file.path(taxdump_dir, "names.dmp")

if (!file.exists(nodes_path)) {
  stop("ERROR: nodes.dmp not found at: ", nodes_path)
} else {
  cat("Found: nodes.dmp\n")
}
if (!file.exists(names_path)) {
  stop("ERROR: names.dmp not found at: ", names_path)
} else {
  cat("Found: names.dmp\n")
}

cat("Reading nodes.dmp...\n")
nodes_dmp <- read.delim(nodes_path, header = FALSE, sep = "|",
                        stringsAsFactors = FALSE, quote = "")
phylum_taxids <- trimws(nodes_dmp[trimws(nodes_dmp$V3) == "phylum", "V1"])
cat("Phylum-rank taxids found:", length(phylum_taxids), "\n")

cat("Reading names.dmp...\n")
names_dmp <- read.delim(names_path, header = FALSE, sep = "|",
                        stringsAsFactors = FALSE, quote = "")
phylum_names <- trimws(names_dmp[
  trimws(names_dmp$V1) %in% phylum_taxids &
    trimws(names_dmp$V4) == "scientific name",
  "V2"
])
cat("Authoritative phylum names extracted:", length(phylum_names), "\n\n")

# ============================================================================
# EXTRACT PHYLUM: Parse ncbi_taxonomy column in input_csv
# ============================================================================

# Normalise ID column name to 'ID' for consistent joining
names(df)[names(df) == "process_id"] <- "ID"

# Also normalise fastp concat sample_name to ID
names(df_fastp_concat)[names(df_fastp_concat) == "sample_name"] <- "ID"

extract_phylum <- function(lineage_str, phylum_names) {
  if (is.na(lineage_str) || lineage_str == "") return(NA_character_)
  elements <- trimws(strsplit(lineage_str, ";")[[1]])
  match    <- elements[elements %in% phylum_names]
  if (length(match) == 0) return(NA_character_)
  match[1]
}

df$phylum <- vapply(df$ncbi_taxonomy, extract_phylum,
                    character(1), phylum_names = phylum_names)

cat("Phylum extraction summary:\n")
print(table(df$phylum, useNA = "ifany"))
cat("\n")

unmatched_phylum <- sum(is.na(df$phylum))
if (unmatched_phylum > 0) {
  cat("WARNING:", unmatched_phylum, "rows with no phylum match\n")
  cat("Affected IDs:\n")
  print(df$ID[is.na(df$phylum)])
  cat("\n")
} else {
  cat("All rows successfully assigned a phylum.\n\n")
}

# ============================================================================
# EXTRACT ORDER: Parse ncbi_taxonomy column using taxdump order-rank names
# ============================================================================
order_taxids <- trimws(nodes_dmp[trimws(nodes_dmp$V3) == "order", "V1"])
cat("Order-rank taxids found:", length(order_taxids), "\n")

order_names <- trimws(names_dmp[
  trimws(names_dmp$V1) %in% order_taxids &
    trimws(names_dmp$V4) == "scientific name",
  "V2"
])
cat("Authoritative order names extracted:", length(order_names), "\n\n")

extract_order <- function(lineage_str, order_names) {
  if (is.na(lineage_str) || lineage_str == "") return(NA_character_)
  elements <- trimws(strsplit(lineage_str, ";")[[1]])
  match    <- elements[elements %in% order_names]
  if (length(match) == 0) return(NA_character_)
  match[1]
}

df$order <- vapply(df$ncbi_taxonomy, extract_order,
                   character(1), order_names = order_names)

cat("Order extraction summary:\n")
print(table(df$order, useNA = "ifany"))
cat("\n")

unmatched_order <- sum(is.na(df$order))
if (unmatched_order > 0) {
  cat("WARNING:", unmatched_order, "rows with no order match\n")
} else {
  cat("All rows successfully assigned an order.\n\n")
}

# ============================================================================
# PLOT 5d: before_total_reads by phylum - boxplot + jitter (MultiQC + static PNG)
# ============================================================================

# Join phylum onto fastp concat by ID
cat("Columns in df_fastp_concat:", paste(names(df_fastp_concat), collapse = ", "), "\n")
cat("Columns in df:", paste(names(df), collapse = ", "), "\n")
cat("Sample IDs in df_fastp_concat (first 5):", paste(head(df_fastp_concat$ID, 5), collapse = ", "), "\n")
cat("Sample IDs in df (first 5):", paste(head(df$ID, 5), collapse = ", "), "\n")

df_phylum_reads <- df_fastp_concat %>%
  dplyr::left_join(df %>% dplyr::select(ID, phylum), by = "ID")

cat("Rows after join:", nrow(df_phylum_reads), "\n")
cat("Rows with phylum NA:", sum(is.na(df_phylum_reads$phylum)), "\n")
cat("Rows with before_total_reads NA:", sum(is.na(df_phylum_reads$before_total_reads)), "\n")

df_phylum_reads <- df_phylum_reads %>%
  dplyr::filter(!is.na(phylum) & !is.na(before_total_reads))

cat("Samples with phylum + read count data:", nrow(df_phylum_reads), "\n")

if (nrow(df_phylum_reads) == 0) {
  cat("WARNING: No samples remain after join - skipping phylum plot.\n")
  cat("Check that ID values match between df_fastp_concat and df.\n")
} else {
  unmatched_join <- sum(is.na(df_phylum_reads$phylum))
  if (unmatched_join > 0) {
    cat("WARNING:", unmatched_join, "samples could not be matched to a phylum\n")
  } else {
    cat("All samples successfully joined to phylum.\n\n")
  }
  
  # Static PNG: boxplot + jitter coloured by phylum
  reads_max      <- max(df_phylum_reads$before_total_reads, na.rm = TRUE)
  reads_break    <- max(1000000, ceiling(reads_max / 10000000) * 10000000)
  n_phyla        <- length(unique(df_phylum_reads$phylum))
  plot_width     <- max(8, n_phyla * 1.5)
  plot_height    <- 6
  
  p_phylum_reads <- ggplot(df_phylum_reads,
                           aes(x = phylum, y = before_total_reads, fill = phylum)) +
    geom_boxplot(alpha = 0.7, outlier.shape = NA) +
    geom_jitter(aes(color = phylum), width = 0.2, alpha = 0.7, size = 2.5) +
    labs(x = "Phylum", y = "Before total reads") +
    theme_minimal() +
    theme(axis.text.x  = element_text(angle = 45, hjust = 1, size = 13),
          axis.text.y  = element_text(size = 13),
          axis.title.x = element_text(size = 13),
          axis.title.y = element_text(size = 13),
          legend.position = "none") +
    scale_y_continuous(
      labels = label_number(scale = 1e-6, suffix = "M"),
      breaks = seq(0, reads_break, by = 10000000)
    )
  
  ggsave(file.path(output_dir, "fastp_before_reads_by_phylum.png"),
         plot = p_phylum_reads, width = plot_width, height = plot_height, dpi = 300)
}

# ============================================================================
# READ AND PROCESS: final_metrics CSV
# ============================================================================
df_metrics <- read.csv(final_metrics_csv, stringsAsFactors = FALSE)

cat("final_metrics dimensions:", nrow(df_metrics), "rows,", ncol(df_metrics), "cols\n\n")

# Convert "null" strings to NA
df_metrics[df_metrics == "null"] <- NA

# Coerce all numeric columns
numeric_columns <- c("n_reads_in", "n_reads_aligned", "n_reads_skipped", "ref_length",
                     "cov_min", "cov_max", "cov_avg", "cov_med",
                     "cleaning_removed_human", "cleaning_removed_at",
                     "cleaning_removed_outlier", "cleaning_removed_reference",
                     "cleaning_ambig_bases", "cleaning_cov_percent")

for (col in numeric_columns) {
  if (col %in% colnames(df_metrics)) {
    df_metrics[[col]] <- as.numeric(df_metrics[[col]])
  }
}

cat("n_reads_in summary:\n")
print(summary(df_metrics$n_reads_in))
cat("n_reads_aligned summary:\n")
print(summary(df_metrics$n_reads_aligned))
cat("\n")

# ============================================================================
# PLOT 6: n_reads_in distribution - static PNG only
# ============================================================================

n_reads_in_vals <- df_metrics$n_reads_in[!is.na(df_metrics$n_reads_in)]

# ============================================================================
# PLOT 7: n_reads_aligned distribution - static PNG only
# ============================================================================

n_reads_aligned_vals <- df_metrics$n_reads_aligned[!is.na(df_metrics$n_reads_aligned)]

# ============================================================================
# PLOT 8: Success rate by raw read count bin (MultiQC bargraph + static PNG)
# ============================================================================

# Derive per-sample success from selected column
sample_success <- df_metrics %>%
  mutate(across(where(is.character), trimws)) %>%
  group_by(ID) %>%
  summarise(
    success = as.integer(any(selected == "YES")),
    .groups = "drop"
  )

cat(sprintf("Total unique samples: %d\n", nrow(sample_success)))
cat(sprintf("Successful barcodes:  %d (%.1f%%)\n",
            sum(sample_success$success),
            100 * mean(sample_success$success)))

# Join fastp read counts
sample_success <- sample_success %>%
  dplyr::left_join(
    df_fastp_concat %>% dplyr::select(ID, before_total_reads),
    by = "ID"
  )

unmatched_fastp <- sum(is.na(sample_success$before_total_reads))
if (unmatched_fastp > 0) {
  cat("WARNING:", unmatched_fastp, "samples have no fastp read count match\n")
} else {
  cat("All samples matched to fastp read counts.\n")
}

# Bin read counts
bin_breaks <- c(0, 5000, 10000, 50000, 100000, 500000, 1000000, 3000000, 5000000,
                8000000, 10000000, 15000000, 20000000, 25000000, 30000000, 35000000,
                40000000, 45000000, 50000000, 55000000, 60000000, Inf)

bin_labels <- c("0-5k", "5k-10k", "10k-50k", "50k-100k", "100k-500k", "500k-1M",
                "1M-3M", "3M-5M", "5M-8M", "8M-10M", "10M-15M", "15M-20M",
                "20M-25M", "25M-30M", "30M-35M", "35M-40M", "40M-45M", "45M-50M",
                "50M-55M", "55M-60M", "60M+")

read_bins <- sample_success %>%
  dplyr::filter(!is.na(before_total_reads)) %>%
  dplyr::mutate(
    read_bin = cut(before_total_reads,
                   breaks         = bin_breaks,
                   labels         = bin_labels,
                   include.lowest = TRUE)
  ) %>%
  dplyr::group_by(read_bin) %>%
  dplyr::summarise(
    n_samples    = n(),
    n_success    = sum(success),
    success_rate = mean(success),
    .groups      = "drop"
  ) %>%
  dplyr::filter(!is.na(read_bin)) %>%
  dplyr::mutate(read_bin = droplevels(read_bin))

cat("\nSuccess rate by read count bin:\n")
print(read_bins)
cat("\n")

# MultiQC YAML - prefix bin labels with zero-padded index for correct sort order
n_bins    <- nrow(read_bins)
pad_width <- nchar(as.character(n_bins))
bin_keys  <- paste0(
  formatC(seq_len(n_bins), width = pad_width, flag = "0"),
  " | ",
  as.character(read_bins$read_bin)
)

mqc_success_bins <- list(
  id                 = "success_rate_by_read_bin",
  parent_id          = "barcoding_outcome",
  parent_name        = "Barcoding Outcome",
  description        = "Barcode success rate by raw read count bin.",
  plot_type          = "bargraph",
  pconfig            = list(
    id           = "success_rate_by_read_bin_bargraph",
    title        = "Success Rate by Read Count Bin",
    ylab         = "Success rate (%)",
    ymax         = 100,
    cpswitch     = FALSE,
    sort_samples = FALSE
  ),
  data = setNames(
    lapply(read_bins$success_rate * 100,
           function(x) list("Success rate (%)" = round(x, 1))),
    bin_keys
  )
)

yaml::write_yaml(mqc_success_bins,
                 file.path(output_dir, "barcoding_03_success_rate_by_read_bin_mqc.yaml"))
cat("Written: barcoding_03_success_rate_by_read_bin_mqc.yaml\n")

# ============================================================================
# VALIDATION PLOTS: Data preparation
# ============================================================================

# df_selected: rows where selected == YES
df_selected <- df_metrics %>%
  mutate(across(where(is.character), trimws)) %>%
  dplyr::filter(selected == "YES")

cat("df_selected rows:", nrow(df_selected), "\n")

# Mode type classification (six categories: {concat,merge,se} x {pre,post-clean})
mode_order_levels <- c("concat", "fcleaner_concat",
                       "merge",  "fcleaner_merge",
                       "se",     "fcleaner_se")

# Whitelist of valid mge_params strings, generated from the r/s grid x the six
# mode tags. Concat is now explicit ("_concat"); legacy unmarked concat strings
# (e.g. "r_1_s_50") are no longer produced and are intentionally excluded.
.r_vals       <- c("r_1", "r_1.3", "r_1.5")
.s_vals       <- c("s_50", "s_100")
.mode_tags    <- mode_order_levels
.base_params  <- as.vector(t(outer(.r_vals, .s_vals, paste, sep = "_")))
mge_params_list <- as.vector(t(outer(.base_params, .mode_tags,
                                      function(b, m) paste0(b, "_", m))))

group_colours <- c(
  "concat"          = "#7BCCC4",
  "fcleaner_concat" = "#FCA082",
  "merge"           = "#DEBADA",
  "fcleaner_merge"  = "#9ECAE1",
  "se"              = "#A1D99B",
  "fcleaner_se"     = "#BCBDDC"
)

# ============================================================================
# VALIDATION PLOT 1 (p1): pident histogram
# ============================================================================
# Computed in static PNGs section below

# ============================================================================
# VALIDATION PLOT 2-1 (p1_stacked): barcode rank stacked by mode type
# ============================================================================

data_param_groups <- df_metrics %>%
  mutate(across(where(is.character), trimws)) %>%
  dplyr::filter(barcode_rank %in% 1:6,
                mge_params %in% mge_params_list) %>%
  dplyr::mutate(
    param_type_grouped = dplyr::case_when(
      grepl("_fcleaner_concat$", mge_params) ~ "fcleaner_concat",
      grepl("_fcleaner_merge$",  mge_params) ~ "fcleaner_merge",
      grepl("_fcleaner_se$",     mge_params) ~ "fcleaner_se",
      grepl("_concat$",          mge_params) ~ "concat",
      grepl("_merge$",           mge_params) ~ "merge",
      grepl("_se$",              mge_params) ~ "se",
      TRUE                                   ~ "concat"
    ),
    param_type_grouped = factor(param_type_grouped, levels = mode_order_levels)
  )

# Complete all rank × mode combinations so absent modes show as 0
rank_param_counts <- data_param_groups %>%
  dplyr::count(barcode_rank, param_type_grouped) %>%
  tidyr::complete(
    barcode_rank       = 1:6,
    param_type_grouped = factor(mode_order_levels, levels = mode_order_levels),
    fill               = list(n = 0L)
  )

rank_totals <- rank_param_counts %>%
  dplyr::group_by(barcode_rank) %>%
  dplyr::summarise(total = sum(n), .groups = "drop")

cat("Barcode rank × mode type counts:\n")
print(rank_param_counts)
cat("\n")

# MultiQC YAML for plot 2-1: stacked bargraph, one entry per barcode rank
mqc_rank_by_mode <- list(
  id                 = "barcode_rank_by_mode",
  parent_id          = "validation",
  parent_name        = "Validation",
  parent_description = "Barcode structural and taxonomic validation metrics.",
  description        = paste0(
    "Number of barcode consensus sequences by rank and preprocessing mode.<br>",
    "<br>",
    "<b>Rank 1</b>: Perfect sequences: no original ambiguous bases; no stop codons; correct reading frame; ≥500 informative bases.<br>",
    "<b>Rank 2</b>: High quality, slightly shorter: no original ambiguous bases; no stop codons; correct reading frame; 400–499 informative bases.<br>",
    "<b>Rank 3</b>: Good quality, moderate length: no original ambiguous bases; no stop codons; correct reading frame; 300–399 informative bases.<br>",
    "<b>Rank 4</b>: Acceptable quality, shorter: no original ambiguous bases; no stop codons; correct reading frame; 200–299 informative bases.<br>",
    "<b>Rank 5</b>: Minimal acceptable quality: no original ambiguous bases; no stop codons; correct reading frame; 1–199 informative bases.<br>",
    "<b>Rank 6</b>: Problematic sequences: contains original ambiguous bases; may have translation issues."
  ),
  plot_type          = "bargraph",
  pconfig            = list(
    id           = "barcode_rank_by_mode_bargraph",
    title        = "Barcode Rank by Mode Type",
    ylab         = "Number of barcode consensus sequences",
    cpswitch     = FALSE,
    sort_samples = FALSE
  ),
  data = setNames(
    lapply(sort(unique(rank_param_counts$barcode_rank)), function(rank) {
      row <- rank_param_counts %>% dplyr::filter(barcode_rank == rank)
      setNames(
        as.list(row$n),
        as.character(row$param_type_grouped)
      )
    }),
    paste0("Rank ", sort(unique(rank_param_counts$barcode_rank)))
  )
)

yaml::write_yaml(mqc_rank_by_mode,
                 file.path(output_dir, "validation_01_barcode_rank_by_mode_mqc.yaml"))
cat("Written: validation_01_barcode_rank_by_mode_mqc.yaml\n")

# ============================================================================
# VALIDATION PLOT 4-1 (p4_1): selected barcodes by mode type
# ============================================================================

selected_by_group <- df_metrics %>%
  mutate(across(where(is.character), trimws)) %>%
  dplyr::filter(mge_params %in% mge_params_list) %>%
  dplyr::mutate(
    mode_type = dplyr::case_when(
      grepl("_fcleaner_concat$", mge_params) ~ "fcleaner_concat",
      grepl("_fcleaner_merge$",  mge_params) ~ "fcleaner_merge",
      grepl("_fcleaner_se$",     mge_params) ~ "fcleaner_se",
      grepl("_concat$",          mge_params) ~ "concat",
      grepl("_merge$",           mge_params) ~ "merge",
      grepl("_se$",              mge_params) ~ "se",
      TRUE                                   ~ "concat"
    )
  ) %>%
  dplyr::group_by(mode_type) %>%
  dplyr::summarise(
    selected_count = sum(selected == "YES", na.rm = TRUE),
    total_count    = n(),
    .groups        = "drop"
  ) %>%
  tidyr::complete(
    mode_type = mode_order_levels,
    fill      = list(selected_count = 0L, total_count = 0L)
  ) %>%
  dplyr::mutate(mode_type = factor(mode_type, levels = mode_order_levels)) %>%
  dplyr::arrange(mode_type)

cat("Selected barcodes by mode type:\n")
print(selected_by_group)
cat("\n")

# MultiQC YAML for plot 4-1
sel_y_ceiling <- max(100, ceiling(max(selected_by_group$selected_count) / 100) * 100)

mqc_selected_by_mode <- list(
  id          = "selected_barcodes_by_mode",
  parent_id   = "validation",
  parent_name = "Validation",
  description = "Number of selected barcodes per preprocessing mode type.",
  plot_type   = "bargraph",
  pconfig     = list(
    id           = "selected_barcodes_by_mode_bargraph",
    title        = "Selected Barcodes by Mode Type",
    ylab         = "Number of selected barcodes",
    ymax         = sel_y_ceiling,
    cpswitch     = FALSE,
    sort_samples = FALSE
  ),
  data = setNames(
    lapply(selected_by_group$selected_count,
           function(n) list("Selected barcodes" = n)),
    as.character(selected_by_group$mode_type)
  )
)

yaml::write_yaml(mqc_selected_by_mode,
                 file.path(output_dir, "validation_02_selected_by_mode_mqc.yaml"))
cat("Written: validation_02_selected_by_mode_mqc.yaml\n")

# ============================================================================
# VALIDATION PLOT 8 (p_order): success rate by taxonomic order
# ============================================================================

# Join order from sequence_references onto sample_success by ID
sample_success_order <- sample_success %>%
  dplyr::left_join(
    df %>% dplyr::select(ID, order) %>% dplyr::distinct(),
    by = "ID"
  )

unmatched_order_join <- sum(is.na(sample_success_order$order))
if (unmatched_order_join > 0) {
  cat("WARNING:", unmatched_order_join, "samples have no order match\n")
} else {
  cat("All samples matched to order.\n")
}

order_success <- sample_success_order %>%
  dplyr::filter(!is.na(order) & order != "") %>%
  dplyr::group_by(order) %>%
  dplyr::summarise(
    n_specimens  = n(),
    n_success    = sum(success),
    success_rate = mean(success),
    .groups      = "drop"
  ) %>%
  dplyr::filter(n_specimens >= 1) %>%
  dplyr::arrange(success_rate) %>%
  dplyr::mutate(order = factor(order, levels = order))

cat("Orders found:", nrow(order_success), "\n\n")

# ============================================================================
# STATIC PNGs
# ============================================================================
outcome_colours <- c(
  "Pass"    = "#2ecc71",
  "Partial" = "#e67e22",
  "Fail"    = "#e74c3c"
)

rank_counts_plot <- rank_counts %>%
  mutate(taxonomic_rank = factor(taxonomic_rank, levels = rev(rank_order)))

p_taxrank <- ggplot(rank_counts_plot, aes(x = taxonomic_rank, y = count)) +
  geom_bar(stat = "identity", fill = "darkgreen", alpha = 0.9) +
  geom_text(aes(label = count), hjust = -0.2, size = 4) +
  theme_minimal() +
  labs(x = "Taxonomic rank", y = "Number of protein pseudo-references") +
  theme(axis.text  = element_text(size = 15),
        axis.title = element_text(size = 15)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  coord_flip()

ggsave(file.path(output_dir, "gene_fetch_taxrank.png"),
       plot = p_taxrank, width = 10, height = 6, dpi = 300)

p_outcome <- ggplot(outcome_counts,
                    aes(x = outcome_category, y = count,
                        fill = outcome_category)) +
  geom_bar(stat = "identity", alpha = 0.9) +
  geom_text(aes(label = count), vjust = -0.3, size = 5) +
  scale_fill_manual(values = outcome_colours) +
  theme_minimal() +
  labs(x = "Outcome", y = "Number of samples") +
  theme(axis.text       = element_text(size = 13),
        axis.title      = element_text(size = 13),
        legend.position = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))

ggsave(file.path(output_dir, "barcoding_outcome.png"),
       plot = p_outcome, width = 8, height = 6, dpi = 300)

p_insert_density <- ggplot(df_fastp_concat, aes(x = insert_size_peak)) +
  geom_density(alpha = 0.6, fill = "steelblue") +
  theme_minimal() +
  labs(x = "Insert size peak (bp)", y = "Density") +
  theme(axis.text  = element_text(size = 13),
        axis.title = element_text(size = 13))

ggsave(file.path(output_dir, "fastp_insert_size_density.png"),
       plot = p_insert_density, width = 8, height = 6, dpi = 300)

p_dup_rate <- ggplot(df_fastp_concat, aes(x = duplication_rate)) +
  geom_histogram(alpha = 0.9, fill = "forestgreen", bins = 100) +
  theme_minimal() +
  labs(x = "Duplication rate", y = "Number of samples") +
  theme(axis.text.x  = element_text(size = 13),
        axis.text.y  = element_text(size = 13),
        axis.title.x = element_text(size = 13),
        axis.title.y = element_text(size = 13)) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))

ggsave(file.path(output_dir, "fastp_duplication_rate.png"),
       plot = p_dup_rate, width = 8, height = 6, dpi = 300)

# Plot 1a: n_reads_in violin + boxplot overlay
df_plot_reads_in <- df_metrics[!is.na(df_metrics$n_reads_in), ]

p_reads_in <- ggplot(df_plot_reads_in, aes(x = "", y = n_reads_in)) +
  geom_violin(fill = "steelblue", alpha = 0.6) +
  geom_boxplot(width = 0.1, fill = "steelblue", alpha = 0.8, outlier.shape = NA) +
  labs(x = "", y = "Number of reads input to MGE") +
  theme_minimal() +
  theme(axis.text.x        = element_blank(),
        axis.ticks.x       = element_blank(),
        panel.grid.major.y = element_line(color = "gray90", linewidth = 0.5),
        panel.grid.major.x = element_blank(),
        axis.text.y        = element_text(size = 15),
        axis.title.y       = element_text(size = 15)) +
  scale_y_continuous(
    labels = label_number(scale = 1e-6, suffix = "M"),
    breaks = pretty(c(0, max(df_plot_reads_in$n_reads_in, na.rm = TRUE)), n = 8)
  )

ggsave(file.path(output_dir, "mge_reads_in_violin.png"),
       plot = p_reads_in, width = 6, height = 6, dpi = 300)

# Plot 3: n_reads_aligned histogram with violin inlay
df_plot_aligned <- df_metrics[!is.na(df_metrics$n_reads_aligned), ]

aligned_max    <- max(df_plot_aligned$n_reads_aligned, na.rm = TRUE)
aligned_hist_y <- max(hist(df_plot_aligned$n_reads_aligned,
                           breaks = 800, plot = FALSE)$counts) * 1.1
inlay_y_max    <- max(8000, ceiling(aligned_max / 1000) * 1000)

p_aligned_inlay <- ggplot(df_plot_aligned, aes(x = "", y = n_reads_aligned)) +
  geom_violin(fill = "darkred", alpha = 0.7) +
  labs(x = NULL, y = NULL) +
  theme_minimal() +
  theme(axis.text.x        = element_blank(),
        axis.ticks.x       = element_blank(),
        axis.text.y        = element_text(size = 8),
        panel.background   = element_rect(fill = "white", color = "black", linewidth = 0.5)) +
  scale_y_continuous(
    labels = label_number(scale = 1e-3, suffix = "k"),
    breaks = pretty(c(0, inlay_y_max), n = 4)
  ) +
  coord_cartesian(ylim = c(0, inlay_y_max))

# Materialise the inset as a grob under a null PDF device. ggplotGrob() needs an
# active graphics device to measure its grobs; in a non-interactive Rscript run
# with no device open it would otherwise spawn a stray Rplots.pdf in the working
# directory. pdf(NULL) absorbs this with no file written.
pdf(NULL)
inlay_grob <- ggplotGrob(p_aligned_inlay)
invisible(dev.off())

p_reads_aligned <- ggplot(df_plot_aligned, aes(x = n_reads_aligned)) +
  geom_histogram(bins = 800, fill = "darkred", alpha = 0.7) +
  theme_minimal() +
  labs(x = "Number of reads aligned to pseudo-reference",
       y = "Frequency") +
  scale_x_continuous(
    labels = label_number(scale = 1e-3, suffix = "k"),
    breaks = pretty(c(0, aligned_max), n = 10)
  ) +
  scale_y_continuous(breaks = pretty(c(0, aligned_hist_y), n = 6)) +
  annotation_custom(
    grob = inlay_grob,
    xmin = aligned_max * 0.65,
    xmax = aligned_max * 1.0,
    ymin = aligned_hist_y * 0.6,
    ymax = aligned_hist_y * 1.05
  )

ggsave(file.path(output_dir, "mge_reads_aligned_histogram_inlay.png"),
       plot = p_reads_aligned, width = 10, height = 6, dpi = 300)

# Plot 8: Success rate by read count bin
p_success_bins <- ggplot(read_bins, aes(x = read_bin, y = success_rate * 100)) +
  geom_bar(stat  = "identity",
           fill  = "steelblue",
           color = "black",
           alpha = 0.9) +
  geom_text(aes(label = n_samples),
            vjust = -0.5,
            color = "black",
            size  = 5) +
  scale_y_continuous(breaks = seq(0, 100, by = 25)) +
  coord_cartesian(ylim = c(0, 110)) +
  labs(x = "Raw read count bin",
       y = "Success rate (%)") +
  theme_minimal() +
  theme(axis.text.x  = element_text(angle = 45, hjust = 1, size = 15),
        axis.text.y  = element_text(size = 15),
        axis.title.x = element_text(size = 15),
        axis.title.y = element_text(size = 15))

ggsave(file.path(output_dir, "barcoding_success_rate_by_read_bin.png"),
       plot = p_success_bins, width = 14, height = 6, dpi = 300)

# ============================================================================
# VALIDATION STATIC PNGs
# ============================================================================

# Validation plot 1: pident histogram
pident_y_max <- max(30, ceiling(max(
  hist(df_selected$pident, breaks = 30, plot = FALSE)$counts
) / 50) * 50)

p_pident <- ggplot(df_selected, aes(x = pident)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black", alpha = 0.7) +
  geom_vline(xintercept = 80, linetype = "dashed", color = "red",    linewidth = 1) +
  geom_vline(xintercept = 90, linetype = "dashed", color = "orange", linewidth = 1) +
  geom_vline(xintercept = 97, linetype = "dashed", color = "green",  linewidth = 1) +
  labs(x = "Percent identity (pident)", y = "Number of samples") +
  theme_minimal() +
  theme(axis.text.x  = element_text(size = 15),
        axis.text.y  = element_text(size = 15),
        axis.title.x = element_text(size = 15),
        axis.title.y = element_text(size = 15)) +
  scale_y_continuous(breaks = pretty(c(0, pident_y_max), n = 8)) +
  coord_cartesian(ylim = c(0, pident_y_max * 1.05))

ggsave(file.path(output_dir, "validation_pident_histogram.png"),
       plot = p_pident, width = 6, height = 6, dpi = 300)

# Validation plot 2-1: barcode rank stacked by mode type
rank_y_max <- max(rank_totals$total, na.rm = TRUE)
rank_y_ceiling <- max(1000, ceiling(rank_y_max / 1000) * 1000)

p_rank_stacked <- ggplot(rank_param_counts,
                         aes(x = factor(barcode_rank), y = n,
                             fill = param_type_grouped)) +
  geom_bar(stat = "identity", color = "black", alpha = 0.9) +
  geom_text(data = rank_totals,
            aes(x = factor(barcode_rank), y = total, label = total, fill = NULL),
            vjust = -0.5, color = "black", size = 4) +
  labs(x = "Barcode rank",
       y = "Number of barcode consensus sequences",
       fill = "Mode type") +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal() +
  theme(legend.position  = "bottom",
        legend.title     = element_text(size = 13),
        axis.text.x      = element_text(size = 15),
        axis.text.y      = element_text(size = 15),
        axis.title.x     = element_text(size = 15),
        axis.title.y     = element_text(size = 15)) +
  scale_y_continuous(
    labels = label_number(scale = 1e-3, suffix = "k"),
    breaks = pretty(c(0, rank_y_ceiling), n = 8)
  ) +
  coord_cartesian(ylim = c(0, rank_y_ceiling * 1.08))

ggsave(file.path(output_dir, "validation_barcode_rank_by_mode.png"),
       plot = p_rank_stacked, width = 6, height = 6, dpi = 300)

# Validation plot 4-1: selected barcodes by mode type
p_selected_mode <- ggplot(selected_by_group,
                          aes(x = mode_type, y = selected_count, fill = mode_type)) +
  geom_bar(stat = "identity", color = "black", alpha = 0.9) +
  geom_text(aes(label = selected_count), vjust = -0.5, color = "black", size = 4) +
  labs(x = "Mode type", y = "Number of selected barcodes") +
  scale_fill_manual(values = group_colours) +
  theme_minimal() +
  theme(axis.text.x      = element_text(size = 15),
        axis.text.y      = element_text(size = 15),
        axis.title.x     = element_text(size = 15),
        axis.title.y     = element_text(size = 15),
        legend.position  = "none") +
  coord_cartesian(ylim = c(0, sel_y_ceiling)) +
  scale_y_continuous(breaks = seq(0, sel_y_ceiling, by = 100))

ggsave(file.path(output_dir, "validation_selected_by_mode.png"),
       plot = p_selected_mode, width = 6, height = 6, dpi = 300)

# Validation plot 8: success rate by taxonomic order
overall_success_rate <- mean(sample_success$success)

n_orders     <- nrow(order_success)
order_height <- max(6, n_orders * 0.35)

p_order_success <- ggplot(order_success, aes(x = order, y = success_rate)) +
  geom_col(aes(fill = success_rate), show.legend = FALSE) +
  geom_hline(yintercept = overall_success_rate,
             linetype = "dashed", color = "black", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.1f%%", 100 * success_rate)),
            hjust = -0.1, size = 4) +
  geom_text(aes(label = sprintf("n=%d", n_specimens)),
            hjust = 1.1, size = 4, color = "white") +
  coord_flip() +
  scale_fill_gradientn(
    colours = c("#e74c3c", "#f39c12", "#2ecc71"),
    values  = if (min(order_success$success_rate) == max(order_success$success_rate))
                c(0, 0.5, 1)
              else
                scales::rescale(c(min(order_success$success_rate),
                                  overall_success_rate,
                                  max(order_success$success_rate)))
  ) +
  scale_y_continuous(
    labels = scales::percent,
    limits = c(0, 1.15),
    breaks = seq(0, 1, by = 0.25)
  ) +
  labs(x = "Taxonomic order",
       y = "Success rate (%)") +
  theme_minimal() +
  theme(axis.text.x  = element_text(size = 13),
        axis.text.y  = element_text(size = 13),
        axis.title.x = element_text(size = 13),
        axis.title.y = element_text(size = 13))

ggsave(file.path(output_dir, "validation_success_rate_by_order.png"),
       plot = p_order_success, width = 14, height = order_height, dpi = 300)

cat("Written: static PNG fallbacks\n")
