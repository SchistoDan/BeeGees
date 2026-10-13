# Human contamination reference

`beegees/resources/contaminants/human_reference.fasta` is the default reference that
`01_human_mitogenome_filter.py` maps reads against (with `bwa aln`) to remove human
contamination. It is used whenever `fasta_cleaner.human_reference` is `null`.

The reference combines the human mitochondrial genome with human NUMTs (nuclear
mitochondrial DNA segments: copies of mitochondrial DNA inserted into the nuclear genome).
A mitogenome on its own only catches mitochondrial contamination. Reads that come from
NUMTs look mitochondrial but differ from the mitogenome, so many of them survive a
mitogenome-only filter. Including the NUMTs lets the filter remove those reads as well.

## Contents
| Sequences | Total length | Min length | Mean length | Max length |
|---|---|---|---|---|
| 769 | 614,536 bp | 52 bp | 799.1 bp | 16,569 bp |

- **1 x human mitogenome:** [NC_012920.1](https://www.ncbi.nlm.nih.gov/nuccore/NC_012920.1) (revised Cambridge Reference Sequence, 16,569 bp)
- **768 x human NUMTs** from [MANUDB](https://github.com/balintbiro/MANUDB/)

NUMT headers follow the format `Homo_sapiens_<MANUDB ID>|<nuclear accession>|<mitochondrial genes covered>`, for example:
```
>Homo_sapiens_61829|NC_000001.11|ND1,TRNI,TRNI,TRNQ,TRNQ,TRNM,TRNM,ND2,TR
```

## Construction process
1. Downloaded the human mitogenome, NC_012920.1, from NCBI.
2. Downloaded all *Homo sapiens* NUMT sequences from [MANUDB](https://github.com/balintbiro/MANUDB/).
3. Concatenated the mitogenome and the NUMTs into a single FASTA (`human_mitogenome-manudb_numts.fasta`).
4. Removed duplicate sequences with `seqkit rmdup`. `-s` compares by sequence rather than
   by ID, and `-i` ignores case, so identical sequences in different cases count as duplicates:
   ```
   seqkit rmdup -s -i human_mitogenome-manudb_numts.fasta > human_mitogenome-manudb_numts-dedup.fasta
   ```
5. Summarised the result with `seqkit stats` (table above):
   ```
   seqkit stats human_mitogenome-manudb_numts-dedup.fasta
   ```
6. Renamed the file to `human_reference.fasta` for packaging.

## Using a different reference
Set `fasta_cleaner.human_reference` in `config.yaml` to the path of any FASTA file.
The `bwa_index_human_ref` rule indexes it once per run, writing the index to
`resources/bwa_index/`. Set `keep_bwa_index: true` to keep the index between runs
when using a large (e.g. genome-scale) reference.
