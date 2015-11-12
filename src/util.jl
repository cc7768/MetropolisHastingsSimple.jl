#=
This function calculates the `cs_percent` credible set for the data
that is provided. It does this by sorting the data and picking out the
appropriate elements on each tail
=#
function credible_set(data::Array{Float64, 1}, cs_percent::Float64)

    # Find how much mass we would like in the tails
    if 0 < cs_percent < 1
        tail_mass = 1. - cs_percent
    elseif 0 < cs_percent < 100
        tail_mass = 1. - cs_percent/100
    else
        print("Error in specifying cs_percent. Returning 90% C.S")
        tail_mass = .1
    end

    # Want to start from mean and work out until we have cs_percent C.S.
    n = length(data)
    sorted_data = sort(data)
    lb = floor(Int, (tail_mass / 2) * n)
    ub = floor(Int, (1. - tail_mass/2) * n)

    return (sorted_data[lb], sorted_data[ub])
end

# From econforge post by Sebastien Villemot
function hp_filter(y::Vector{Float64}, lambda::Float64=1600.)
    n = length(y)
    @assert n >= 4

    diag2 = lambda*ones(n-2)
    diag1 = [ -2lambda; -4lambda*ones(n-3); -2lambda ]
    diag0 = [ 1+lambda; 1+5lambda; (1+6lambda)*ones(n-4); 1+5lambda; 1+lambda ]

    D = spdiagm((diag2, diag1, diag0, diag1, diag2), (-2,-1,0,1,2))

    trend = D\y
    cycle = y-trend

    return cycle, trend
end

# Take a mean and std dev and give the IG parameters that correspond
function IG_params(mean::Float64, std::Float64)

    alpha = (std/mean)^2 + 2
    beta = mean*(alpha-1)

    return alpha, beta
end