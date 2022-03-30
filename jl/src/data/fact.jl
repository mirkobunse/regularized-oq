"""
    struct Fact <: DataSet

    Fact([configfile = "conf/data/fact.yml"; kwargs...])

FACT telescope data set.

**Keyword arguments:**

- `readdata=true` whether to load the data or produce a dummy instance
- `nobs=typemax(Int)` how many observations to read
- `num_bins=-1` the number of bins (or -1 for the value in the `configfile`)
"""
struct Fact <: DataSet
    configfile::String
    X_data::Matrix{Float32} # sklearn handles Float32
    y_data::Vector{Float32}
    discretizer::LinearDiscretizer

    function Fact(configfile::String="conf/data/fact.yml"; nobs::Union{Integer, Nothing}=nothing,
                  readdata::Bool=true, num_bins::Int=-1, mode::Symbol=:default, kwargs...)
        if length(kwargs) > 0 # just warn, do not throw an error
            @warn "Unused keyword arguments in data configured by $configfile" kwargs...
        end
        X_data, y_data = _Xy_fact(configfile, nobs, readdata, mode)
        d = LinearDiscretizer(configfile; (num_bins > 0 ? [:num_bins=>num_bins] : [])...)
        return new(configfile, X_data, y_data, d)
    end
end

# implementation of interface
X_data(d::Fact) = d.X_data
y_data(d::Fact) = d.y_data
discretizer(d::Fact) = d.discretizer

# read FACT data
function _Xy_fact(configfile::String, nobs::Union{Integer, Nothing}, readdata::Bool, mode::Symbol)
    X = zeros(Float32, 0, 20)
    y = zeros(Float32, 0)
    if readdata
        c = parsefile(configfile; dicttype=Dict{Symbol,Any})
        if mode == :default
            mode = Symbol(c[:mode])
        end
        files = String[] # file paths to read from
        if mode ∈ [:wobble, :all]
            push!(files, c[:wobble])
        end
        if mode ∈ [:diffuse, :all]
            push!(files, c[:diffuse])
        end
        for datafile in files
            df = DataFrames.disallowmissing!(CSV.read(datafile, DataFrame; limit=nobs))
            y = vcat(y, Vector{Float32}(df[:, FACT_TARGET]))
            X = vcat(X, Matrix{Float32}(df[:, setdiff(propertynames(df), [FACT_TARGET])]))
        end
    end
    return X, y
end

FACT_TARGET = :log10_energy

"""
    get_fact(configfile="conf/data/fact.yml")

Download the FACT data from the URLs specified in the `configfile` and prepare this data
for the experiments.
"""
get_fact(configfile::String="conf/data/fact.yml") = mktempdir() do tmpdir
    c = parsefile(configfile; dicttype=Dict{Symbol, Any})
    for mode in [:wobble, :diffuse]
        url = c[:download][mode]
        tmp = joinpath(tmpdir, basename(url)) # local temporary path

        @info "Downloading $(url) to $(tmpdir)"
        download(url, tmp) # from the Base package
        df = _read_fact_hdf5(tmp) # process the downloaded file

        @info "Writing prepared data to $(c[mode])"
        CSV.write(c[mode], df)
    end
end

# sub-routine creating a meaningful DataFrame from a full HDF5 file
function _read_fact_hdf5(path::String)
    df = DataFrame()

    # read each HDF5 groups that astro-particle physicists read according to
    # https://github.com/fact-project/open_crab_sample_analysis/blob/f40c4fab57a90ee589ec98f5fe3fdf38e93958bf/configs/aict.yaml#L30
    features = [
        :size,
        :width,
        :length,
        :skewness_trans,
        :skewness_long,
        :concentration_cog,
        :concentration_core,
        :concentration_one_pixel,
        :concentration_two_pixel,
        :leakage1,
        :leakage2,
        :num_islands,
        :num_pixel_in_shower,
        :photoncharge_shower_mean,
        :photoncharge_shower_variance,
        :photoncharge_shower_max,
    ]
    h5open(path, "r") do file
        for f in [
                :corsika_event_header_total_energy, # the target variable
                :cog_x, # needed only for feature generation; will be removed
                :cog_y,
                features...
                ]
            df[!, f] = read(file, "events/$(f)")
        end
    end

    # generate additional features, according to
    # https://github.com/fact-project/open_crab_sample_analysis/blob/f40c4fab57a90ee589ec98f5fe3fdf38e93958bf/configs/aict.yaml#L50
    df[!, :log_size] = log.(df[!, :size])
    df[!, :area] = df[!, :width] .* df[!, :length] .* π
    df[!, :size_area] = df[!, :size] ./ df[!, :area]
    df[!, :cog_r] = sqrt.(df[!, :cog_x].^2 + df[!, :cog_y].^2)

    # the label needs to be transformed for log10 plots
    df[!, FACT_TARGET] = log10.(df[!, :corsika_event_header_total_energy])

    # convert the label and the actual features to Float32, to find non-finite elements
    df = df[!, [FACT_TARGET, :log_size, :area, :size_area, :cog_r, features...]]
    for column in names(df)
        df[!, column] = convert.(Float32, df[!, column])
    end

    # only return instances without NaNs (by pseudo broadcasting)
    return filter(row -> all([ isfinite(cell) for cell in row ]), df)
end
