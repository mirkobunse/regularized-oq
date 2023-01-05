module Util

using CSV, CherenkovDeconvolution, DataFrames, PyCall, ScikitLearn

"""
    SkObject(class_name, args...; kwargs...)

This compilation-ready version of `ScikitLearn.@sk_import` calls the constructor of the
fully-qualified `class_name` with the given `args` and `kwargs`.
"""
function SkObject(class_name::AbstractString, args...; kwargs...)
    # @info "The import uses scikit-learn $(pyimport(split(class_name, ".")[1]).__version__)"
    Constructor = getproperty(
        pyimport(join(split(class_name, ".")[1:end-1], ".")), # package name
        Symbol(split(class_name, ".")[end]) # un-qualified class name
    )
    return Constructor(args...; kwargs...)
end

"""
    numpy_seterr(; kwargs...) = np.seterr(; kwargs...)

Set how numpy handles floating-point errors.
"""
numpy_seterr(; kwargs...) = pyimport("numpy").seterr(; kwargs...)

"""
    rnod(a, b)

Root Normalized Order-aware Divergence (RNOD) [sakai2021evaluating].
"""
rnod(a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number =
    sqrt(__od(a, b) / (length(a) - 1))

"""
    rsnod(a, b)

Root Symmetric Normalized Order-aware Divergence (RSNOD) [sakai2021evaluating].
"""
rsnod(a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number =
    sqrt((__od(a, b)/2 + __od(b, a)/2) / (length(a) - 1))

function __od(a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number
    __check_distance_arguments(a, b)
    d = (a - b) .^ 2 # (p_j - p*_j)^2 for all j
    DW = i -> sum(abs(i - j) * d[j] for j in 1:length(b)) # Eq. 12 in [sakai2021evaluating]
    C_star = findall(b .> 0) # C*, the classes with non-zero probability
    return sum(DW, C_star) / length(C_star) # Eq. 13 in [sakai2021evaluating]
end

__check_distance_arguments(a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number =
    if length(a) != length(b)
        throw("length(a) = $(length(a)) != length(b) = $(length(b))")
    elseif !isapprox(sum(a), sum(b))
        throw("histograms have to have the same mass (difference is $(sum(a)-sum(b))")
    end

"""
    nmd(a, b) = mdpa(a, b) / (length(a) - 1)

Compute the Normalized Match Distance (NMD) [sakai2021evaluating], a variant of the Earth
Mover's Distance [rubner1998metric] which is normalized by the number of classes.
"""
nmd(a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number =
    mdpa(a, b) / (length(a) - 1)

"""
    mdpa(a, b)

Minimum Distance of Pair Assignments (MDPA) [cha2002measuring] for ordinal pdfs `a` and `b`.
The MDPA is a special case of the Earth Mover's Distance [rubner1998metric] that can be
computed efficiently.
"""
function mdpa(a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number
    __check_distance_arguments(a, b)
    prefixsum = 0.0 # algorithm 1 in [cha2002measuring]
    distance  = 0.0
    for i in 1:length(a)
        prefixsum += a[i] - b[i]
        distance  += abs(prefixsum)
    end
    return distance / sum(a) # the normalization is a fix to the original MDPA
end

end # module
