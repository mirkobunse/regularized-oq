__precompile__(false)
module DeconvExp

import Conda, VersionParsing

function __init__()
    conda_list = Conda.parseconda(`list scikit-learn`)
    sk_version = length(conda_list) > 0 ? conda_list[1]["version"] : "999"
    if VersionParsing.vparse(sk_version) != v"1.0.2"
        Conda.runconda(`install -y scikit-learn=1.0.2`)
        error("scikit-learn successfully installed; you need to restart Julia now.")
    end # fix the scikit-learn version for reproducibility
end

export Util, Data, MoreMethods, Conf, Job, Res

include("Util.jl")
using .Util

include("Data.jl")
using .Data

include("MoreMethods.jl")
using .MoreMethods

include("Conf.jl")
using .Conf

include("Job.jl")
using .Job

include("Res.jl")
using .Res

end # module
