# Cohort v1 audit

Run date: 2026-10-22. The [inclusion policy](cohort-policy.md) was frozen before model training. Its canonical SHA-256 is `6452bbc7e3bbd8fb1cba3f92afe120b076677966695739020560163a563793fb`.

## Metadata screening

The source snapshot contains 1,112 participants. The table assigns each excluded participant to the first applicable reason in policy order, so exclusions do not overlap.

| Stage | Excluded at this stage | Remaining |
| --- | ---: | ---: |
| Published metadata | 0 | 1,112 |
| Derivative identifier unavailable | 77 | 1,035 |
| Missing or invalid mean FD | 1 | 1,034 |
| Mean FD at or above 0.2 mm | 150 | 884 |
| At least one manual QC rating is not OK | 346 | 538 |
| Acquisition group below 20 participants or 10 per diagnosis | 104 | 434 |

No additional participants were removed for invalid diagnosis, site, age, or published sex code after the earlier exclusions. The eligibility count is 434, comprising 205 autism and 229 control participants. Original site labels number 10; paired samples collapse into seven acquisition groups.

| Acquisition group | Autism | Control | Total |
| --- | ---: | ---: | ---: |
| LEUVEN | 12 | 20 | 32 |
| NYU | 65 | 82 | 147 |
| TRINITY | 17 | 19 | 36 |
| UCLA | 23 | 19 | 42 |
| UM | 39 | 62 | 101 |
| USM | 32 | 17 | 49 |
| YALE | 17 | 10 | 27 |
| Total | 205 | 229 | 434 |

Among metadata-eligible participants, age ranges from 6.47 to 50.22 years (mean 16.41). Mean FD averages 0.0743 mm. The source records 354 participants with sex code 1 and 80 with code 2. Neither diagnosis groups nor acquisition groups are equally sized.

| Metadata-eligible diagnosis group | Mean age, years | Mean FD, mm | Sex code 2 |
| --- | ---: | ---: | ---: |
| Autism | 16.62 | 0.0862 | 26 / 205 (12.7%) |
| Control | 16.23 | 0.0636 | 54 / 229 (23.6%) |

Residual motion and sex composition differ between diagnosis groups even after screening. The first connectivity-only baseline leaves those associations in the data. Later nuisance and demographic controls must assess their contribution before interpreting a score difference as a disease-related connectivity effect.

The metadata-eligible set represents 41.9% of the 1,035 participants with derivative identifiers. Strict manual QC and the minimum diagnosis counts remove several acquisition groups, including PITT. The original NYU/PITT smoke test therefore remains an integration check separate from this research cohort. Results from this selected population cannot be treated as results for all ABIDE participants.

## Scan validation

The full build validated all 434 selected scans. Two NYU control scans contained a constant ROI signal and were excluded. Both had 176 timepoints, so the scan-length rule did not cause their exclusion. No additional acquisition group fell below its minimum.

The final cohort contains 432 participants: 205 autism and 227 control participants across seven acquisition groups. NYU contributes 145 participants (65 autism, 80 control); the other group counts in the metadata table are unchanged. Retained scans contain 116 to 296 timepoints and 200 ordered ROI columns. Each participant contributes 19,900 finite Fisher-z edges.

The final age mean is 16.36 years, and mean FD is 0.0744 mm. There are 353 participants with sex code 1 and 79 with code 2. The control group now has mean age 16.12 years, mean FD 0.0638 mm, and 53 participants with sex code 2 (23.3%). Autism-group summaries are unchanged. The final set represents 41.7% of participants with derivative identifiers.

A separate offline rebuild produced identical arrays, manifests, and exclusion decisions using the cached files and receipts. Local outputs are under `data/cohort/v1/`; the offline verification outputs are under `data/cohort/v1-offline/`.

| Fingerprint | SHA-256 |
| --- | --- |
| Source metadata | `009f01a89697113e5375ded4fbcff4c85e89a4daa6d13e63c21ae5d57dda0865` |
| Final research manifest | `02275091d0a8fe1edbbd27c39ef91b12f522f0fd683c09fb5e63eb5504869caa` |
| First research dataset archive | `65e4243398a81058e5ec57b3654e571a2f71653328ac0fbdae05f4f0e7eb64b0` |

The archive fingerprint identifies this file. Array equality is the offline reproduction check; archive bytes can differ if a NumPy or archive-format implementation changes.
