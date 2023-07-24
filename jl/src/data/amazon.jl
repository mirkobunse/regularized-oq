"""
    struct AmazonRoberta <: DataSet

    AmazonRoberta([kwargs...])

Amazon roberta embeddings.

**Keyword arguments:**

- `X_path="/mnt/data/amazon-oq-bk/roberta/unique_samples_X.npy"` where to read features from.
- `y_path="/mnt/data/amazon-oq-bk/roberta/unique_samples_y.npy"` where to read labels from.
- `readdata=true` whether to load the data or produce a dummy instance
"""
struct AmazonRoberta <: DataSet
    X::Matrix{Float32} # sklearn handles Float32
    y::Vector{Int32}
    function AmazonRoberta(;
            X_path::String="/mnt/data/amazon-oq-bk/roberta/unique_samples_X.npy",
            y_path::String="/mnt/data/amazon-oq-bk/roberta/unique_samples_y.npy",
            readdata::Bool=true,
            kwargs...
            )
        if length(kwargs) > 0 # just warn, do not throw an error
            @warn "Unused keyword arguments in data configured by $configfile" kwargs...
        end
        X = zeros(Float32, 0, 768)
        y = zeros(Int64, 0)
        if readdata
            np = pyimport("numpy")
            X = np.load(X_path)
            y = np.load(y_path) .+ 1
        end
        return new(X, y)
    end
end

# implementation of interface
X_data(d::AmazonRoberta) = d.X
y_data(d::AmazonRoberta) = d.y
discretizer(d::AmazonRoberta) = CategoricalDiscretizer(Dict(zip(1:5, 1:5))) # dummy mapping
