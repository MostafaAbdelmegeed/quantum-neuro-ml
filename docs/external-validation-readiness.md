# ABIDE II external validation readiness

Reviewed 2026-10-22. The metadata inventory now contains 1,007 ABIDE II candidates, with no detected numeric ID overlap against all 1,112 ABIDE I metadata identities. This is a partial metadata result. It establishes neither a complete participant crosswalk nor a quality-controlled external cohort. Both successful inventory runs set `model_eligible: false`; no imaging data were downloaded and no ABIDE II model was evaluated.

## Completed metadata inventory

The [inventory script](../scripts/external_metadata_inventory.py) and [frozen configuration](../config/abide2-inventory-v1.json) snapshot public source bytes, URLs, checksums, retrieval times, and missing-file responses. The latest guarded [local report](../data/external/abide2-inventory-v1-audited/report.json) and its [offline reproduction](../data/external/abide2-inventory-v1-reproduced/report.json) record:

| Check | Result |
| --- | --- |
| Listed collection prefixes | 19 |
| Readable participant tables | 16 |
| Metadata candidates | 1,007: 474 ASD and 533 controls |
| Missing participant tables | `ABIDEII-ETHZ_1`, `ABIDEII-UCLA_Long`, `ABIDEII-UPSM_Long` |
| Detected numeric overlap with ABIDE I | 0 rows, comparing all 1,112 development identities |
| Candidate manifest SHA-256 | `75e565699610d971fa712cb96c7129acf0bcf1c6450898183f0dfa639ea43fed` |
| Offline reproduction | Byte-identical `candidates.csv` and identical collection summaries |

Both latest run receipts match their saved report checksums, and their candidate CSV files and collection summaries reproduce exactly. The parser trims headers, validates the development file's unnamed sequential row index, and explicitly handles GU_1's isolated non-breaking-space byte. All 1,007 rows meet the metadata checks; none has passed imaging or motion QC. KUL_3 and NYU_2 contain only ASD candidates, so the metadata count also includes collections unsuitable for a per-site balanced-accuracy estimate on their own.

## Participant independence

The ABIDE II descriptor identifies 1,044 cross-sectional datasets from people whose data were not in ABIDE I. It separately includes 38 returning participants in two longitudinal collections, using their original ABIDE I IDs. Thus, the release name alone does not establish independence. [ABIDE II descriptor, Data Records](https://pmc.ncbi.nlm.nih.gov/articles/PMC5349246/).

The configuration excludes `ABIDEII-UCLA_Long` and `ABIDEII-UPSM_Long` in full, including follow-up scans. Their narratives identify the baseline scans as ABIDE I data. The script canonicalizes numeric IDs across `sub-` prefixes and leading zeros. [Longitudinal collection narratives](https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html), [PCP identifier convention](https://github.com/preprocessed-connectomes-project/abide/blob/gh-pages/download.md).

Neither longitudinal participant table was available at the queried public path. Consequently, `known_longitudinal_rows: 0` means that no such rows entered the readable tables; it does not mean that the release contains no returning participants. The missing tables and ETHZ_1 remain outside the audit. Moreover, 281 candidate rows have a nonmissing GUID hash, but the frozen ABIDE I metadata has no `NDAR_GUID` column. The implemented GUID comparison therefore cannot establish cross-release disjointness from this snapshot. Obtain the official identity mappings and reconcile renamed IDs before calling the eventual cohort participant-independent.

Several ABIDE II collections come from institutions that contributed to ABIDE I. Report validation at returning institutions separately from validation at new institutions; a new participant does not necessarily mean a new scanner or acquisition protocol. [ABIDE II descriptor, Usage Notes](https://pmc.ncbi.nlm.nih.gov/articles/PMC5349246/).

## What is available

| Resource | Verified route and limitation |
| --- | --- |
| Composite phenotypes | The official [cross-sectional CSV](https://www.nitrc.org/frs/downloadlink.php/9108) and [longitudinal CSV](https://www.nitrc.org/frs/downloadlink.php/9109) currently redirect anonymous requests to NITRC login. No login was attempted. |
| Public site metadata | Sixteen anonymously readable tables supply the inventory above. The [NYU_1 participants table](https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE2/RawData/ABIDEII-NYU_1/participants.tsv), for example, has 78 rows and includes participant, site, diagnosis, age, and sex. |
| Public imaging inventory | The [RawData inventory](https://fcp-indi.s3.amazonaws.com/?list-type=2&prefix=data%2FProjects%2FABIDE2%2FRawData%2F&delimiter=%2F) contains site folders, including both longitudinal collections. The inspected listing lacks the later Stanford and Miami collections. Reconcile coverage with the official release before claiming a cohort size. |
| Public derivatives | The [Outputs inventory](https://fcp-indi.s3.amazonaws.com/?list-type=2&prefix=data%2FProjects%2FABIDE2%2FOutputs%2F&delimiter=%2F) contains `denoise`, `fmriprep`, `mindboggle_swf`, and `mriqc`; the [Derivatives inventory](https://fcp-indi.s3.amazonaws.com/?list-type=2&prefix=data%2FProjects%2FABIDE2%2FDerivatives%2F&delimiter=%2F) contains `DCAN`. The producer also [documents the Outputs routes](https://sensein.group/open-data-processing/abide2). These names do not establish compatibility with PCP C-PAC. |

The [phenotype legend](https://fcon_1000.projects.nitrc.org/indi/abide/ABIDEII_Data_Legend.pdf) defines diagnosis as 1=autism, 2=control, and sex as 1=male, 2=female. It includes `NDAR_GUID` where provided; this can support overlap checks but cannot replace the release and ID audit.

Rebuild the candidate inventory from the existing metadata cache without network access; use a new output directory because completed results cannot be overwritten:

```powershell
uv run --locked --extra dev --extra graph --extra quantum python scripts/external_metadata_inventory.py --config config/abide2-inventory-v1.json --cache-dir data/external/abide2-metadata-v1 --output-dir data/external/abide2-inventory-v1-rebuild --offline
```

For a fresh metadata snapshot, omit `--offline` and supply new cache and output directories. Existing cache receipts preserve the original responses, including 404s. INDI documents [anonymous S3 access](https://fcon_1000.projects.nitrc.org/indi/s3/index.html). The specific candidate prefix `data/Projects/ABIDE2/Outputs/cpac/filt_noglobal/rois_cc200/` returned zero keys during this review. This establishes its absence at that path, not the absence of compatible derivatives everywhere.

## Preprocessing and quality gates

The current data use PCP `cpac/filt_noglobal/rois_cc200` time series. The [PCP download specification](https://github.com/preprocessed-connectomes-project/abide/blob/gh-pages/download.md) establishes that layout under `ABIDE_Initiative`, not under `ABIDE2`. Its [C-PAC description](https://github.com/preprocessed-connectomes-project/abide/blob/gh-pages/cpac.md) specifies 24 motion regressors, five tissue components, trends, and optional 0.01-0.1 Hz filtering/global signal regression, but leaves the software release as "version X". A modern C-PAC or fMRIPrep run cannot be called an exact reproduction from those labels alone.

All 16 readable tables lack `func_mean_fd` and all five manual QC fields required by [cohort v1](../config/cohort-v1.json). Missing QC cannot count as passing. Establish a separate external policy with documented motion calculation, manual review, scan selection, minimum timepoints, and atlas label order. Compare it with the development policy before fitting anything.

Release provenance also matters: official corrections cover orientation errors in OHSU/UPSM/GU, duplicated longitudinal scans, BNI timepoint order, and reassigned OHSU IDs. A mirror timestamp alone does not demonstrate that every correction reached every derivative. [September 2016 corrections](https://fcon_1000.projects.nitrc.org/indi/abide/updates/ABIDEII-Usernotes_Updates_Fixes_9_25_16.pdf), [March 2017 corrections](https://fcon_1000.projects.nitrc.org/indi/abide/updates/ABIDEII-Usernotes_Updates_Fixes_3_27_17.pdf).

## Data use and next action

All 16 audited collection descriptions declare Creative Commons Attribution-NonCommercial-Share Alike; [NYU_1's description](https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE2/RawData/ABIDEII-NYU_1/dataset_description.json) is one example. INDI permits [non-commercial use and describes registration requirements](https://fcon_1000.projects.nitrc.org/). Confirm terms for additional collections and assemble consortium/site funding acknowledgements before reporting their results. Cite Di Martino et al. (2017), [doi:10.1038/sdata.2017.10](https://doi.org/10.1038/sdata.2017.10), the dataset [doi:10.15387/FCP_INDI.ABIDE2](https://doi.org/10.15387/FCP_INDI.ABIDE2), and the preprocessing/atlas sources actually used. The paper's CC-BY licence does not override the data terms.

Next, reconcile the missing tables and later release additions against official metadata, obtain the identity crosswalk, and document corrected source provenance. Then resolve a source for compatible CC200 features and an external QC policy. If compatibility cannot be established, choose between reprocessing both releases with one pinned pipeline or declaring a separate test under preprocessing shift. Neither choice belongs in the present frozen runs.

Freeze the external policy and final model list before inspecting ABIDE II performance. Fit residualization, scaling, PCA, encoder, and model selection using ABIDE I only, then apply the locked models once to the external cohort. Until these gates pass, report 1,007 metadata candidates and zero external model evaluations. A retained QC cohort size remains unknown.
