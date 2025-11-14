# Hidden Markov Models of permissable barcoding gene regions

## COI-5P
- Created Sat Mar 28 02:53:28 2020
- Comprised of 38,243 COI-5P sequences
- 658bp in length
### Construction process:
1. **DETERMINE SOURCE OF HMM**

## rbcL barcoding region
- Created Wed Nov 12 13:11:28 2025
- Comprised of 13,580 rbcL barcoding region sequences
- 601bp in length
### Construction process:
1. 441,131 rbcL sequences downloaded from the following sources:
   - Full rbcL database produced in [Dubious et al (2022)](https://bmcgenomdata.biomedcentral.com/articles/10.1186/s12863-022-01067-5), accessible [here](https://figshare.com/articles/online_resource/QIIME2_RefDB_development_zip/17040680?file=56104238) = 378,134 sequences
   - rbcL sequences extracted from the complete BOLD public database (31 October 2025 release) = 24,533 sequences
   - Plant metabarcoding database produced in [Bell et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5357121/), accesible [here](https://figshare.com/articles/dataset/rbcL_July_2021/14936007) = 38,464 sequences
2. All short sequences < 300bp were removed with `seqkit seq`
   - 443,723 sequences remaining
3. All 'illegal' (for downstream tools) gap ('-') characters were replaced with ambiguous base calls ('N')
4. The barcoding region was extracted from the [rbcL reference sequence](https://www.ncbi.nlm.nih.gov/nuccore/NC_001666.2?from=56874&to=58304&report=genbank) using `seqkit amplicon`, employing the rbcLa primer pair outlined in [CBOL Plant Working Group, 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2722355/).
5. Using the same primer pair, the rbcL barcoding region was extracted from the remaining 443,723 sequences
   - 69,999 complete rbcL barcode sequences remaining
6. The 69,999 rbcL barcode sequences were then aligned to the reference sequence using MAFFT
7. The multiple sequence alignment of 70,000 sequences was visualised and manually quality controlled in AliView.
   - Sequences with excess ambiguous bases and those that introduced gaps in the alignment were removed
   - 69,973 sequences remaining
8. Aligned rbcL barcode region sequences were then dereplicated using `vsearch --derep_fulllength`
   - 13,580 non-redundant sequences
9. HMM created using `hmmbuild`
   - **= 601 bp HMM representing the barcode region of rbcL**
