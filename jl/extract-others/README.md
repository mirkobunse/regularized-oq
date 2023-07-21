# UCI and OpenML Data Sets for Ordinal Quantification

These four labeled data sets are targeted at ordinal quantification. The goal of quantification is not to predict the label of each individual instance, but the distribution of labels in unlabeled sets of data.

With the scripts provided, you can extract CSV files from the UCI machine learning repository and from OpenML. The ordinal class labels stem from a binning of a continuous regression label.

We complement this data set with the indices of data items that appear in each sample of our evaluation. Hence, you can precisely replicate our samples by drawing the specified data items. The indices stem from two evaluation protocols that are well suited for ordinal quantification. To this end, each row in the files `app_val_indices.csv`, `app_tst_indices.csv`, `app-oq_val_indices.csv`, and `app-oq_tst_indices.csv` represents one sample.

Our first protocol is the artificial prevalence protocol (APP), where all possible distributions of labels are drawn with an equal probability. The second protocol, APP-OQ, is a variant thereof, where only the smoothest 20% of all APP samples are considered. This variant is targeted at ordinal quantification tasks, where classes are ordered and a similarity of neighboring classes can be assumed.


## Usage

You can extract four CSV files through the provided script `extract-oq.jl`, which is conveniently wrapped in a `Makefile`. The `Project.toml` and `Manifest.toml` specify the Julia package dependencies, similar to a requirements file in Python.

**Preliminaries:** You have to have a working Julia installation. We have used Julia v1.6.5 in our experiments.

**Data Extraction:** In your terminal, you can call either

```
make
```

(recommended), or

```
julia --project="." --eval "using Pkg; Pkg.instantiate()"
julia --project="." extract-oq.jl
```

**Outcome:** The first row in each CSV file is the header. The first column, named "class_label", is the ordinal class.


## Further Reading

Implementation of our experiments: https://github.com/mirkobunse/regularized-oq
