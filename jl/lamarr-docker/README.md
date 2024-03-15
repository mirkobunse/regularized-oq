# Setting up a Julia-ready container

We proceed as follows, starting on *your local computer* (gwkilab does not support the first two steps).

1. initialize a git submodule for building standard Docker images at Lamarr.
2. build the image using parts of the submodule.
3. start a Slurm job in the way that is usual at Lamarr.

## Prerequisites

Before we start, you need to install and generate an API token for NVidia's NGC system, install their `ngc` client locally, and have a working `docker` installation. To meet these prerequisites, follow the general instructions from the [Lamarr cluster documentation](https://gitlab.tu-dortmund.de/lamarr/lamarr-public/cluster#custom-docker-images).

## Building the Docker image

Now, initialize the git submodule at `lamarr-docker/custom-container-example/`:

```
git submodule init
git submodule update
```

Then, build our image. This process takes quite some time.

```
make
```

**Note:** You can always pull the git submodule to receive the latest version of the standard build process. Change the `lamarr-docker/Dockerfile` to customize the Docker image that is being built.

## Starting a Slurm job

You can now start a Slurm job from this image, from gwkilab, as usual. Consider using the `lamarr-docker/srun.sh` script for this purpose.

## Finalizing the installation within the running job

Navigate to the `jl/` directory and instantiate the Julia project from the `Project.toml` and `Manifest.toml` files.

```julia
using Pkg
pkg"activate ."
pkg"instantiate"
pkg"build"
pkg"precompile"

# fix the scikit-learn version
import Conda
Conda.runconda(`install -yc anaconda scipy==1.9.1`)
Conda.runconda(`install -y scikit-learn=1.0.2`)

# install the other Python dependencies
Conda.runconda(`install -y pandas`)
Conda.pip_interop(true)
Conda.pip("install", "git+https://github.com/HLT-ISTI/QuaPy")
```
