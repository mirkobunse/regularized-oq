# Cherenkov Telescope Data for Ordinal Quantification

Bla bla.


## Usage

You extract the data `fact.csv` through the provided script `extract-fact.jl`, which is conveniently wrapped in a `Makefile`. The `Project.toml` and `Manifest.toml` specify the Julia package dependencies, similar to a requirements file in Python. In your terminal, you can call either

```
make
```

or

```
julia --project="." --eval "using Pkg; Pkg.instantiate()"
julia --project="." extract-fact.jl
```

The first row in `fact.csv` is the header. The first column, named "log10_energy", is the continuous target quantity. We map this quantity to ordinal classes in terms of binning its values. By default, we use 12 bins between 2.4 and 4.2. These values represent the logarithm of each gamma ray energy in Giga-Electron Volt, i.e. log10(energy / GeV). Binning these values is conventional practice in astro-particle physics.
