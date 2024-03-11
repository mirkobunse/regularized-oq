using Pkg
@info "Installing package dependencies with setup.jl"
pkg"instantiate"
pkg"build"
pkg"precompile"

# fix the scikit-learn version for reproducibility
using Conda
Conda.runconda(`install -yc anaconda scipy==1.9.1`)
Conda.runconda(`install -y scikit-learn=1.0.2`)

# install the other Python dependencies
Conda.runconda(`install -y pandas`)
Conda.pip_interop(true)
Conda.pip("install", "git+https://github.com/HLT-ISTI/QuaPy")
