# Supplementary material for "Ordinal Quantification Through Regularization"

The `supplement.pdf` contains additional material, e.g., extended experimental results.

The directories `jl/` and `py/` contain the source code of our methods and experiments, as written in Julia and Python, respectively. Please consult their `jl/README.md` and `py/README.md` files for more information.

- the Python code implements the extraction of the Amazon-OQ-BK dataset and the ordinal classifier experiment (Tab. 1 in our paper).
- the Julia code implements the extraction of the FACT-OQ dataset and the comparison experiment (Tab. 2 in our paper).

We use two programming languages because we could build, for the respective tasks, on existing public code. In particular, we would like to thank the authors of [QuaPy](https://github.com/HLT-ISTI/QuaPy), [mord](https://github.com/fabianp/mord), and [CherenkovDeconvolution.jl](https://github.com/mirkobunse/CherenkovDeconvolution.jl) for making their code publicly available.
