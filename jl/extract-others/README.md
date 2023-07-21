# Cherenkov Telescope Data for Ordinal Quantification

This labeled data set is targeted at ordinal quantification. It appears in our research paper "Ordinal Quantification Through Regularization", which we have published at ECML-PKDD 2022. The goal of quantification is not to predict the label of each individual instance, but the distribution of labels in unlabeled sets of data.

With the scripts provided, you can extract the relevant features and labels from the public data set of the FACT Cherenkov telescope. These features are precisely the ones that domain experts from astro-particle physics employ in their analyses. The labels stem from a binning of a continuous energy label, which is common practice in these analyses.

We complement this data set with the indices of data items that appear in each sample of our evaluation. Hence, you can precisely replicate our samples by drawing the specified data items. The indices stem from two evaluation protocols that are well suited for ordinal quantification. To this end, each row in the files `app_val_indices.csv`, `app_tst_indices.csv`, `app-oq_val_indices.csv`, and `app-oq_tst_indices.csv` represents one sample.

Our first protocol is the artificial prevalence protocol (APP), where all possible distributions of labels are drawn with an equal probability. The second protocol, APP-OQ, is a variant thereof, where only the smoothest 20% of all APP samples are considered. This variant is targeted at ordinal quantification tasks, where classes are ordered and a similarity of neighboring classes can be assumed. The labels of the FACT data lie on an ordinal scale and, hence, pose such an ordinal quantification task.


## Usage

You can extract the data `fact.csv` through the provided script `extract-fact.jl`, which is conveniently wrapped in a `Makefile`. The `Project.toml` and `Manifest.toml` specify the Julia package dependencies, similar to a requirements file in Python.

**Preliminaries:** You have to have a working Julia installation. We have used Julia v1.6.5 in our experiments.

**Data Extraction:** In your terminal, you can call either

```
make
```

(recommended), or

```
curl --fail -o fact.hdf5 https://factdata.app.tu-dortmund.de/dl2/FACT-Tools/v1.1.2/gamma_simulations_facttools_dl2.hdf5
julia --project="." --eval "using Pkg; Pkg.instantiate()"
julia --project="." extract-fact.jl
```

**Outcome:** The first row in the resulting `fact.csv` file is the header. The first column, named "class_label", is the ordinal class.


## Further Reading

Implementation of our experiments: https://github.com/mirkobunse/ecml22

Original data repository: https://factdata.app.tu-dortmund.de/

Reference analysis by astro-particle physicists: https://github.com/fact-project/open_crab_sample_analysis
