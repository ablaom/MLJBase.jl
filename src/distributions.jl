## CONSTANTS

const Dist = Distributions


## EQUALITY OF DISTRIBUTIONS (minor type piracy)

function ==(d1::D, d2::D) where D<:Dist.Sampleable
    ret = true
    for fld in fieldnames(D)
        ret = ret && getfield(d1, fld) == getfield(d2, fld)
    end
    return ret
end


# DISTRIBUTION AS TRAIT (needed?)

# fallback:
isdistribution(d) = isdistribution(typeof(d))
isdistribution(::Type{<:Any}) = false

# for anything sampleable in Distributions.jl:
isdistribution(::Type{<:Dist.Sampleable}) = true


# ADD TO distributions.jl TYPE HIERARCHY TO ACCOUNT FOR NON-EUCLIDEAN
# SUPPORTS

abstract type NonEuclidean <: Distributions.ValueSupport end


## UNIVARIATE NOMINAL PROBABILITY DISTRIBUTION

"""
    UnivariateFinite(levels, p)

A discrete univariate distribution whose finite support is the
elements of the vector `levels`, and whose corresponding probabilities
are elements of the vector `p`, which must sum to one.

    UnivariateFinite(prob_given_level)

A discrete univariate distribution whose finite support is the set of
keys of the provided dictionary, `prob_given_level`. The dictionary
values specify the corresponding probabilities, which must be
nonnegative and sum to one. 

    levels(d::UnivariateFinite)

Return the support of `d`. In the special case that
`keys(prob_given_level)` has type `AbstractVector{L}` where `L <:
CategoricalValue` or `L <: CategoricalString` (for example it is a
`CategoricalVector`) the levels returned includes all categorical
values/strings sharing the pool of the values appearing as keys. One
may call `pdf` on all such values, with zero returned if the level
does not appear as a key.

````julia
d = UnivariateFinite(["yes", "no", "maybe"], [0.1, 0.2, 0.7])
pdf(d, "no") # 0.2
mode(d) # "maybe"
rand(d, 5) # ["maybe", "no", "maybe", "maybe", "no"]
d = fit(UnivariateFinite, ["maybe", "no", "maybe", "yes"])
pdf(d, "maybe") ≈ 0.5 # true
levels(d) # ["yes", "no", "maybe"]
````
See also `classes`.

"""
struct UnivariateFinite{L,U,T<:Real} <: Dist.Distribution{Dist.Univariate,NonEuclidean}
    pool::CategoricalPool{L,U}
    prob_given_level::LittleDict{U,T}
end

function UnivariateFinite(prob_given_level::AbstractDict{L}) where L
    L <: CategoricalElement ||
        error("The support of a UnivariateFinite can consist only of "*
              "CategoricalString or CategoricalValue elements. ")
end

function UnivariateFinite(prob_given_level::AbstractDict{L,T}) where {U<:Unsigned,L<:CategoricalElement{U},T<:Real}
    
    an_element = first(keys(prob_given_level))
    pool = an_element.pool
    
    p = values(prob_given_level) |> collect
    Dist.@check_args(UnivariateFinite, Dist.isprobvec(p))
    
    d = LittleDict{U,T}()
    for key in classes(an_element)
        haskey(prob_given_level, key) && (d[key.level] = prob_given_level[key] )
    end
    return UnivariateFinite(pool, d)
end

function UnivariateFinite(levels::AbstractVector{L},
                          p::AbstractVector{<:Real}) where L
    L <: CategoricalElement || error("levels must have CategoricalValue or "*
                                     "CategoricalString type.")
    Dist.@check_args(UnivariateFinite, length(levels)==length(p))
    prob_given_level = LittleDict([levels[i]=>p[i] for i in eachindex(p)])
    return  UnivariateFinite(prob_given_level)
end

# CategoricalArrays.levels(d::UnivariateFinite) = collect(keys(d.prob_given_level))

function classes(d::UnivariateFinite)
    p = d.pool
    return [p.valindex[p.invindex[v]] for v in p.levels]
#    return (d.pool).valindex[sortperm(d.pool.order)]
end

function Distributions.support(d::UnivariateFinite)
    p = d.pool
    return sort(p.valindex[collect(keys(d.prob_given_level))])
end

function Base.show(stream::IO, d::UnivariateFinite)
    support = Dist.support(d)
    x1 = first(support)
    p1 = d.prob_given_level[x1.level]
    str = "UnivariateFinite($x1=>$p1"
    pairs = (x=>d.prob_given_level[x.level] for x in support[2:end])
    for pair in pairs
        str *= ", $(pair[1])=>$(pair[2])"
    end
    str *= ")"
    show(stream, str)
end

function average(dvec::AbstractVector{UnivariateFinite{L,U,T}};
                 weights=nothing) where {L,U,T}

    n = length(dvec)
    
    Dist.@check_args(UnivariateFinite, weights == nothing || n==length(weights))

    # check all distributions have consistent pool:
    first_index = first(dvec).pool.index
    for d in dvec
        d.pool.index == first_index ||
            error("Averaging UnivariateFinite distributions with incompatible"*
                  " pools. ")
    end

    # get all refs:
    refs = reduce(union, [keys(d.prob_given_level) for d in dvec])

    # pad each individual dicts so they have common keys:
    z = LittleDict{U,T}([x => zero(T) for x in refs]...)    
    prob_given_level_vec = map(dvec) do d
        merge(z, d.prob_given_level)
    end

    # initialize the prob dictionary for the distribution sum:
    prob_given_level = LittleDict{U,T}()
    for x in refs
        prob_given_level[x] = zero(T)
    end
    
    # sum up:
    if weights == nothing
        scale = 1/n
        for x in refs
            for k in 1:n
                prob_given_level[x] += scale*prob_given_level_vec[k][x]
            end
        end
    else
        scale = 1/sum(weights)
        for x in refs
            for k in 1:n
                prob_given_level[x] +=
                    weights[k]*prob_given_level_vec[k][x]*scale
            end
        end
    end

    return UnivariateFinite(first(dvec).pool, prob_given_level)
    
end        

function Distributions.mode(d::UnivariateFinite)
    dic = d.prob_given_level
    p = values(dic)
    max_prob = maximum(p)
    m = first(first(dic)) # mode, just some level for now
    for (x, prob) in dic
        if prob == max_prob
            m = x
            break
        end
    end
    return d.pool.valindex[m]
end

function _pdf(d::UnivariateFinite{L,U,T}, ref) where {L,U,T}
    if haskey(d.prob_given_level, ref)
        return d.prob_given_level[ref]
    else
        return zero(T)
    end
end

function Distributions.pdf(d::UnivariateFinite{L,U,T},
                           x::CategoricalElement) where {L,U,T}
    x in classes(d) || throw(ArgumentError(""))
    return _pdf(d, x.level)
end

function Distributions.pdf(d::UnivariateFinite{L,U,T},
                           level::L) where {L,U,T}
    if haskey(d.pool.invindex, level)
        return _pdf(d, d.pool.invindex[level])
    else
        throw(ArgumentError(""))
    end
end


"""
    _cummulative(d::UnivariateFinite)

Return the cummulative probability vector `[0, ..., 1]` for the
distribution `d`, using whatever ordering is used in the dictionary
`d.prob_given_level`. Used only for to implement random sampling from
`d`.

"""
function _cummulative(d::UnivariateFinite{L,U,T}) where {L,U,T<:Real}
    p = collect(values(d.prob_given_level))
    K = length(p)
    p_cummulative = Array{T}(undef, K + 1)
    p_cummulative[1] = zero(T)
    p_cummulative[K + 1] = one(T)
    for i in 2:K
        p_cummulative[i] = p_cummulative[i-1] + p[i-1]
    end
    return p_cummulative
end


"""
_rand(p_cummulative)

Randomly sample the distribution with discrete support `1:n` which has
cummulative probability vector `p_cummulative=[0, ..., 1]` (of length
`n+1`). Does not check the first and last elements of `p_cummulative`
but does not use them either. 

"""
function _rand(p_cummulative)
    real_sample = rand()
    K = length(p_cummulative)
    index = K
    for i in 2:K
        if real_sample < p_cummulative[i]
            index = i - 1
            break
        end
    end
    return index
end

function Base.rand(d::UnivariateFinite)
    p_cummulative = _cummulative(d)
    levels = d.pool.valindex[collect(keys(d.prob_given_level))]
    return levels[_rand(p_cummulative)]
end

function Base.rand(d::UnivariateFinite, n::Int)
    p_cummulative = _cummulative(d)
    levels = d.pool.valindex[collect(keys(d.prob_given_level))]
    return [levels[_rand(p_cummulative)] for i in 1:n]
end

function Distributions.fit(d::Type{<:UnivariateFinite},
                           v::AbstractVector{L}) where L
    L <: CategoricalElement ||
        error("Can only fit a UnivariateFinite distribution to samples of "*
              "CategoricalValue or CategoricalString type. ")
    vpure = skipmissing(v) |> collect
    isempty(vpure) && error("No non-missing data to fit. ")
    N = length(vpure)
    count_given_level = Dist.countmap(vpure)
    prob_given_level = LittleDict([x=>c/N for (x, c) in count_given_level])
    return UnivariateFinite(prob_given_level)
end
    
    
## NORMAL DISTRIBUTION ARITHMETIC

# *(lambda::Number, d::Dist.Normal) = Dist.Normal(lambda*d.μ, lambda*d.σ)
# function +(ds::Dist.Normal...)
#     μ = sum([d.μ for d in ds])
#     σ = sum([d.σ^2 for d in ds]) |> sqrt
#     return Dist.Normal(μ, σ)
# end

# hack for issue #809 in Dist.jl:
Distributions.fit(::Type{Dist.Normal{T}}, args...) where T =
    Dist.fit(Dist.Normal, args...)

