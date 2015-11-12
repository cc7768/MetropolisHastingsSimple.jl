using Distributions
using HDF5

# ------------------------------------------------------------------- #
# Random Walk Metropolis Model
# ----------------------------
# This is just a type that holds useful information such as the number
# of parameters, how to evaluate the log Posterior Kernel, and how to
# propose new values
# ------------------------------------------------------------------- #
immutable RW_Metropolis_Model
    nparams::Int            # Number of parameters estimating
    Post_Kernel::Function   # Function to evaluate posterior kernel (P + LL)
    proposal::Distribution  # Proposal distribution
end

# Constructor that takes variance of steps
function RW_Metropolis_Model(data, Post_Kernel, Σ)
    nparams = size(Σ, 1)
    propdist = MultivariateNormal(zeros(nparams), Σ)
    return RW_Metropolis_Model(nparams, Post_Kernel, propdist)
end

# ------------------------------------------------------------------- #
# Random Walk Metropolis State
# ----------------------------
# This is just a type that holds the current state of the random walk
# Metropolis model. It holds the current state value, the value of the
# posterior kernel at that state, and whether the last proposal was
# accepted or rejected
# ------------------------------------------------------------------- #
type RW_Metropolis_State
    state::Array{Float64, 1}     # Current value of state
    post_kernel::Float64         # The kernel of the posterior
    accept_prev::Int             # Did I accept previous step?
end

#=
This function takes a RWM model and state as its inputs. It proposes
a new value to move the state to and then decides whether to accept the
move or to stay at the current state
=#
function update_state!(rwm::RW_Metropolis_Model, rws::RW_Metropolis_State)
    # Propose a new value using the random walk
    proposal = rws.state + rand(rwm.proposal)

    # Evaluate the proposed value's log posterior kernel
    post_kernel = rwm.Post_Kernel(proposal)

    # Compute the acceptance ratio
    acceptance = post_kernel - rws.post_kernel

    # Update
    if log(rand()) < acceptance
        # Accepted
        rws.state = proposal
        rws.post_kernel = post_kernel
        rws.accept_prev = 1
    else
        # Didn't accept
        rws.accept_prev = 0
    end

    nothing
end


# ------------------------------------------------------------------- #
# Random Walk Metropolis Run
# ------------------------------------------------------------------- #
#=
This is the most general run function. It takes a random walk metropolis
model and an initial start and dispatches the run to either the run
function which returns an array of the samples or to the run function
which saves the results into an hdf file named `fname`
=#
function run_RWM(rwm::RW_Metropolis_Model, rws::RW_Metropolis_State,
                 nsamples::Int; burn::Int=1000, skip::Int=5,
                 hdf_save=false, fname="Samples", iterstosave=25000)

    # Burn it before passing it off
    for t=1:burn
        update_state!(rwm, rws)
    end

    if hdf_save
        run_RWM_hdf(rwm, rws, nsamples; skip=skip, fname=fname,
                    iterstosave=iterstosave)
    else
        return run_RWM_nohdf(rwm, rws, nsamples; skip=skip)
    end

end

#=
This function a random walk Metropolis algorithm for a specified model,
and while it runs, it saves the data into an array which is returned
when the function finishes running.
=#
function run_RWM_nohdf(rwm::RW_Metropolis_Model, rws::RW_Metropolis_State,
                       nsamples::Int; skip::Int=5)

    # Allocate space for my samples
    samples = Array(Float64, rwm.nparams, nsamples)
    niters = nsamples*skip
    fillcounter = 0
    accept_rate = 0


    # Metropolis draws
    # Suggested improvement
    #  for i=1:nsamples
    #     for _=1:skip  # do thinning iterations
    #         update_state(rwm, rws)
    #     end
    #     samples[:, i] = rws.state
    #     accept_rate += rws.accept_prev
    # end

    for iter=1:niters

        update_state!(rwm, rws)

        if iter % skip == 0
            fillcounter += 1
            samples[:, fillcounter] = rws.state
            accept_rate += rws.accept_prev
        end
    end
    println("Acceptance rate was $(accept_rate/nsamples)")
    return samples
end

#=
This function runs a random walk Metropolis algorithm for a specified
model and while it runs, it saves the data into an hdf file which can
be inspected after the function is done
=#
function run_RWM_hdf(rwm::RW_Metropolis_Model, rws::RW_Metropolis_State,
                     nsamples::Int; skip::Int=5,
                     fname::String="fname", iterstosave::Int=25000)

    # Setup
    niters = nsamples*skip

    # Open an HDF5 file and allocate its memory space
    hdf_file = HDF5.h5open(fname, "w")
    hdf_dset = HDF5.d_create(hdf_file, "params", datatype(Float64),
                            dataspace(rwm.nparams, nsamples), "chunk",
                            (rwm.nparams, iterstosave))

    # Create an array that will hold data that hasn't been saved
    samples = Array(Float64, rwm.nparams, iterstosave)
    fillcounter = 0
    savecounter = 0
    accept_rate = 0

    # Metropolis draws
    for iter=1:niters

        update_state!(rwm, rws)

        if iter % skip == 0
            fillcounter += 1
            samples[:, fillcounter] = rws.state
            accept_rate += rws.accept_prev
        end

        if fillcounter == iterstosave
            fillcounter = 0  # Reset the samples counter so we refill data
            savecounter += 1
            lb, ub = (savecounter-1)*iterstosave + 1, savecounter*iterstosave
            hdf_dset[:, lb:ub] = samples
        end
    end

    println("Acceptance rate was $(accept_rate/nsamples)")
    nothing
end
