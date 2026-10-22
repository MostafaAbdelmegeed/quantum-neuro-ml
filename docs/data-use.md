# ABIDE data use and citations

Reviewed on 2026-10-22 for this non-commercial methodological study.

The [ABIDE I usage agreement](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html) permits non-commercial research and links the Creative Commons Attribution-NonCommercial-ShareAlike licence. It asks users to identify the datasets analysed and acknowledge their funding sources. Its raw-data route requires NITRC and FCP/INDI registration. The separate [PCP download documentation](https://github.com/preprocessed-connectomes-project/abide/blob/gh-pages/download.md) describes public S3 access to preprocessed derivatives; this project uses that route.

The repository's code licence does not apply to downloaded data. Keep participant metadata, ROI series, matrices, and predictions outside Git. Reports should identify retained acquisition groups, cite the dataset and preprocessing sources, and include the applicable consortium and site funding acknowledgements from the ABIDE site profiles. Do not attempt participant identification.

Required research references include:

- Di Martino et al. (2014), *The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism*. Molecular Psychiatry. [doi:10.1038/mp.2013.78](https://doi.org/10.1038/mp.2013.78).
- Craddock et al. (2013), *The Neuro Bureau Preprocessing Initiative: open sharing of preprocessed neuroimaging data and derivatives*. Neuroinformatics 2013. [PCP citation instructions](https://github.com/preprocessed-connectomes-project/abide/blob/gh-pages/publications.md).
- The C-PAC pipeline, as requested by the [PCP project](https://preprocessed-connectomes-project.org/index.html), with its [ABIDE preprocessing configuration](https://github.com/preprocessed-connectomes-project/abide/blob/gh-pages/cpac.md).
- Craddock et al. (2012), *A whole brain fMRI atlas generated via spatially constrained spectral clustering*. Human Brain Mapping. [doi:10.1002/hbm.21333](https://doi.org/10.1002/hbm.21333).

The historical C-PAC documentation leaves its release version unspecified. Record the published derivative configuration and source URLs without inventing a software version. Before a paper or redistributed derivative release, assemble the funding acknowledgements for the actual included sites and recheck the linked terms.
