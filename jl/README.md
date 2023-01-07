# Experiments in Julia

This project consists of the following parts:

- scripts to generate a Docker image
- Julia code to generate the FACT-OQ data in Julia, see `src/data/fact.jl`
- changeable YAML configurations for our comparison experiment (Tab. 2), see `conf/meta/`
- Julia code to run experiments from the YAML configurations `src/Experiments.jl`
- Julia code to generate Tab. 2 and additional tables from the experimental results, see `src/Results.jl`
- Unit tests to ensure our implementations are correct, see `test/`

We recommend you to set up a Docker container, to have an isolated and controlled environment for running the experiments. However, you can also run them in your local Julia installation.

## Docker setup

Build the image and start a container. You will get a terminal inside the container, from where you can go on with the next section.

```
cd docker/
make

# if you also push to a repository if the environment variable ${DOCKER_REPOSITORY} has its URL
make push

# or you can specify a different group name, group ID, user name, or user ID
make GID=1234 GROUP=foo UID=5678 USER=bar

# once the build has finished, you can start a container
./run.sh
```

The `run.sh` script will mount two directories: your home folder `/home/<user>/` will be mounted at `/mnt/home/` and a data directory (which can be configured and is `/rdata/s01f_c3_004/` by default) will be mounted at `/mnt/data/`. The data directory should contain a sub-directory `amazon-oq-bk`, where the Amazon-OQ-BK data is stored.

## Setup inside the container / local setup without Docker

Navigate to the place where you have stored our supplementary material. Start `julia` from the `jl/` directory. You get an interactive session with which you can generate the FACT-OQ data set.

```
julia> using Revise, OrdinalQuantification
julia> Data.get_fact()
```

From this interactive session, you can also update the configuration files in `conf/gen/` after having changed the meta-configurations in `conf/meta/`.

```
julia> Configuration.amazon()
julia> Configuration.dirichlet()
```

## Running experiments

The individual experiments are configured in `conf/gen/`, the contents of which are automatically generated from a more high-level meta-configuration in `conf/meta/`.

You can run an experiment either from the interactive Julia shell (recommended only for testing):

```
julia> Experiments.run("conf/gen/test_<experiment>.yml")
```

Or you can run multiple experiments in batch mode with multiple cores (recommended for actual experimentation). Call from a regular shell (not from Julia):

```
julia -p <number of cores> main.jl [--no-validate] conf/gen/<file 1>.yml conf/gen/<file 2>.yml ...
```

If you use a **local setup** without Docker, the above command has to be changed to

```
julia --project=. -p <number of cores> main.jl [--no-validate] conf/gen/<file 1>.yml conf/gen/<file 2>.yml ...
```

We have run all experiments on 40 cores, with which each experiment took between 2 and 12 hours. Testing is much faster, only a few minutes on a single core.

Essentially, an experiment maps a configuration from `conf/gen/` to a results file in `res/csv/`.

## Generating tables

You first need to prepare the results in `res/csv/` by running the experiments (see above).

In the Julia shell, you can call

```
julia> Results.main()

# the supplementary comparison table with other data sets is generated through
julia> Results.main("res/tex/main_others.tex"; metricsfiles=Results.METRICSFILES_OTHERS)

# the "ranking" tables for each individual data set are generated through
julia> Results.ranking("res/csv/<file>.csv")
```

Essentially, these calls map a results file in `res/csv/` to LaTeX code in `res/tex/`.
