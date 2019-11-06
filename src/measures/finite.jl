## CLASSIFICATION METRICS (FOR DETERMINISTIC PREDICTIONS)

struct MisclassificationRate <: Measure end

"""
    misclassification_rate(ŷ, y)
    misclassification_rate(ŷ, y, w)

Returns the rate of misclassification of the (point) predictions `ŷ`,
given true observations `y`, optionally weighted by the weights
`w`. All three arguments must be abstract vectors of the same length.

For more information, run `info(misclassification_rate)`.

"""
misclassification_rate = MisclassificationRate()
name(::Type{<:MisclassificationRate}) = "misclassification_rate"
target_scitype(::Type{<:MisclassificationRate}) = AbstractVector{<:Finite}
prediction_type(::Type{<:MisclassificationRate}) = :deterministic
orientation(::Type{<:MisclassificationRate}) = :loss
reports_each_observation(::Type{<:MisclassificationRate}) = false
is_feature_dependent(::Type{<:MisclassificationRate}) = false
supports_weights(::Type{<:MisclassificationRate}) = true

(::MisclassificationRate)(ŷ::AbstractVector{<:CategoricalElement},
                          y::AbstractVector{<:CategoricalElement}) =
                              mean(y .!= ŷ)
(::MisclassificationRate)(ŷ::AbstractVector{<:CategoricalElement},
                          y::AbstractVector{<:CategoricalElement},
                          w::AbstractVector{<:Real}) =
                              sum((y .!= ŷ) .*w) / sum(w)


## CLASSIFICATION METRICS (FOR PROBABILISTIC PREDICTIONS)

struct CrossEntropy <: Measure end

"""
    cross_entropy(ŷ, y::AbstractVector{<:Finite})

Given an abstract vector of `UnivariateFinite` distributions `ŷ` (ie,
probabilistic predictions) and an abstract vector of true observations
`y`, return the negative log-probability that each observation would
occur, according to the corresponding probabilistic prediction.

For more information, run `info(cross_entropy)`.

"""
cross_entropy = CrossEntropy()
name(::Type{<:CrossEntropy}) = "cross_entropy"
target_scitype(::Type{<:CrossEntropy}) = AbstractVector{<:Finite}
prediction_type(::Type{<:CrossEntropy}) = :probabilistic
orientation(::Type{<:CrossEntropy}) = :loss
reports_each_observation(::Type{<:CrossEntropy}) = true
is_feature_dependent(::Type{<:CrossEntropy}) = false
supports_weights(::Type{<:CrossEntropy}) = false

# for single observation:
_cross_entropy(d, y) = -log(pdf(d, y))

function (::CrossEntropy)(ŷ::AbstractVector{<:UnivariateFinite},
                          y::AbstractVector{<:CategoricalElement})
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)
    return broadcast(_cross_entropy, Any[ŷ...], y)
end

# TODO: support many distributions/samplers D below:
struct BrierScore{D} <: Measure end

"""
    brier = BrierScore(; distribution=UnivariateFinite)
    brier(ŷ, y)

Given an abstract vector of distributions `ŷ` and an abstract vector
of true observations `y`, return the corresponding Brier (aka
quadratic) scores.

Currently only `distribution=UnivariateFinite` is supported, which is
applicable to superivised models with `Finite` target scitype. In this
case, if `p(y)` is the predicted probability for a *single*
observation `y`, and `C` all possible classes, then the corresponding
Brier score for that observation is given by

``2p(y) - \\left(\\sum_{η ∈ C} p(η)^2\\right) - 1``

For more information, run `info(brier_score)`.

"""
function BrierScore(; distribution=UnivariateFinite)
    distribution == UnivariateFinite ||
        error("Only `UnivariateFinite` Brier scores currently supported. ")
    return BrierScore{distribution}()
end

name(::Type{<:BrierScore{D}}) where D = "BrierScore{$(string(D))}"
target_scitype(::Type{<:BrierScore{D}}) where D = AbstractVector{<:Finite}
prediction_type(::Type{<:BrierScore}) = :probabilistic
orientation(::Type{<:BrierScore}) = :score
reports_each_observation(::Type{<:BrierScore}) = true
is_feature_dependent(::Type{<:BrierScore}) = false
supports_weights(::Type{<:BrierScore}) = true

# For single observations (no checks):

# UnivariateFinite:
function brier_score(d::UnivariateFinite, y)
    levels = classes(d)
    pvec = broadcast(pdf, d, levels)
    offset = 1 + sum(pvec.^2)
    return 2*pdf(d, y) - offset
end

# For multiple observations:

# UnivariateFinite:
function (::BrierScore{<:UnivariateFinite})(
    ŷ::AbstractVector{<:UnivariateFinite},
    y::AbstractVector{<:CategoricalElement})
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)
    return broadcast(brier_score, Any[ŷ...], y)
end

function (score::BrierScore{<:UnivariateFinite})(
    ŷ, y, w::AbstractVector{<:Real})
    check_dimensions(y, w)
    return w .* score(ŷ, y) ./ (sum(w)/length(y))
end


# Confusion matrix (for any number of classes) and associated
# metrics in the binary case
