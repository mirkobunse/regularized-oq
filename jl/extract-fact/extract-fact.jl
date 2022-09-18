#
# julia extract-fact.jl
#
using CSV, DataFrames, Discretizers, HDF5

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
    df[!, :log10_energy] = log10.(df[!, :corsika_event_header_total_energy])

    # map the continuous target quantity to ordinal classes
    discr = LinearDiscretizer(range(2.4, stop=4.2, length=13)) # 12 bin representation
    df[!, :class_label] = encode(discr, df[!, :log10_energy])

    # convert the features (but not the label) to Float32, to find non-finite elements
    df = df[!, [:class_label, :log_size, :area, :size_area, :cog_r, features...]]
    for column in names(df)[2:end]
        df[!, column] = convert.(Float32, df[!, column])
    end

    # only return instances without NaNs (by pseudo broadcasting)
    return filter(row -> all([ isfinite(cell) for cell in row ]), df)
end

# extract the FACT data from fact.hdf5 to fact.csv
@info "Reading from fact.hdf5"
df = _read_fact_hdf5("fact.hdf5") # process the downloaded file
@info "Writing prepared data to fact.csv"
CSV.write("fact.csv", df)
