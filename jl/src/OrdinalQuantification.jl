module OrdinalQuantification

import Conda, VersionParsing

function __init__()
    conda_list = Conda.parseconda(`list scikit-learn`)
    sk_version = length(conda_list) > 0 ? conda_list[1]["version"] : "999"
    if VersionParsing.vparse(sk_version) != v"1.0.2"
        Conda.runconda(`install -y scikit-learn=1.0.2`)
        @warn "scikit-learn successfully installed; you might need to restart Julia now"
    end # fix the scikit-learn version for reproducibility
end

export Util, Data, MoreMethods, Configuration, Experiments, Results

include("Util.jl")
using .Util

include("Data.jl")
using .Data

include("MoreMethods.jl")
using .MoreMethods

include("Configuration.jl")
using .Configuration

include("Experiments.jl")
using .Experiments

include("Results.jl")
using .Results

end # module
