CASTANO_DATASET_NAMES = [
    "abalone.ord_chu",
    "ailerons_gago",
    "auto.data.ord_chu",
    "bostonhousing.ord_chu",
    "californiahousing_gago",
    "cement_strength_gago",
    "ERA",
    "ESL",
    "kinematics_gago",
    "LEV",
    "SkillCraft1_rev_7clases",
    "SkillCraft1_rev_8clases",
    "skill_gago",
    "stock.ord",
    "SWD",
    "winequality-red_gago",
    "winequality-white_gago_rev",
]

castano_url(dataset_name::String) = "https://raw.githubusercontent.com/mirkobunse/ordinal_quantification/main/datasets/ordinal/$(dataset_name).csv"

"""
    struct CastanoDataSet <: DataSet

A type for datasets used by Castano et al. (2022).

**See also:** `X_data`, `y_data`, `discretizer`
"""
struct CastanoDataSet <: DataSet
    X_data::Matrix{Float32} # sklearn handles Float32
    y_data::Vector{Int32}
    function CastanoDataSet(dataset_name::String)
        df = CSV.read(Downloads.download(castano_url(dataset_name)), DataFrame)
        X_data = Matrix{Float32}(df[:,1:end-1])
        y_data = Int32.(df[!,end])
        return new(X_data, y_data .- minimum(y_data) .+ 1)
    end
end

# implementation of the interface
X_data(d::CastanoDataSet) = d.X_data
y_data(d::CastanoDataSet) = d.y_data
function discretizer(d::CastanoDataSet) # create a dummy / identity mapping
    bins_y = 1:maximum(d.y_data)
    return CategoricalDiscretizer(Dict(zip(bins_y, bins_y)))
end
