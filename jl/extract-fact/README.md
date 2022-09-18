# Cherenkov Telescope Data for Ordinal Quantification

This labeled data set is well suited to evaluate ordinal quantification methods. It appears in our research paper "Ordinal Quantification Through Regularization", which we have published at ECML-PKDD 2022. The goal of quantification is not to predict the label of each individual instance, but the distribution of labels in unlabeled sets of data. This Cherenkov telescope data is continuously labeled, as in regression. However, domain experts from astro-particle physics conventionally map these values to bins, which correspond to ordered classes.

With the scripts provided, you can extract the relevant features from the public FACT telescope data set. These features are precisely the ones that domain experts from astro-particle physics employ in their analyses.

We complement this data with the indices of data items that appear in each sample of our evaluation. Hence, you can reproduce our samples by drawing exactly the specified data items, which stem from two evaluation protocols for (ordinal) quantification. Alternatively, you can implement these protocols yourself or implement additional evaluation protocols.

Our first protocol is the artificial prevalence protocol (APP), where all possible distributions of labels are drawn with an equal probability. The second protocol, APP-OQ, is a variant thereof, where only the smoothest 20% of all APP samples are considered. This variant is targeted at ordinal quantification, where classes are ordered and a similarity of neighboring classes can be assumed. The labels of the FACT data lie on an ordinal scale and, hence, pose such an ordinal quantification task.

Implementation of our experiments: https://github.com/mirkobunse/ecml22

Original data repository: https://factdata.app.tu-dortmund.de/

Reference analysis by astro-particle physicists: https://github.com/fact-project/open_crab_sample_analysis


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
