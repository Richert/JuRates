using DifferentialEquations,Plots,Statistics,FileIO,Random,JLD2

# definition of fixed parameters
τ = 1.0
τ_d = 10.0
Δ = 2.0
J = 15.0*√Δ
J_in = 0.0
η_b = -5.22
N = 1000
prob = 0.1
prob_in = 0.2
v_th = 10000.0
α = 0.05

# lorenz system parameters
u0_lorenz = [1.0,0.0,0.0]
p_lorenz = [10.0,28.0,8/3]

# simulation parameters
tspan = [0., 20.]
dts = 0.01

# definition of cortical input
# stim_t = 50.0
# stim_v = 8.0
# stim_off = 8.0
# stim_cut = 10.0
# ctx_t = truncated(Normal(stim_t, stim_v), stim_t - stim_cut, stim_t + stim_cut)
# str_t = truncated(Normal(stim_t+stim_off, stim_v), stim_t+stim_off-stim_cut, stim_t+stim_off+stim_cut)

# definition of the equations of motion
function qif_exp_lorenz(du, u, p, t)

    # extract state vars and params
	###############################

    v, a, ls = u[1:N], u[N+1:2*N], u[2*N+1:2*N+3]
    τ, η, C, v_th, τ_d, α, lp, W, J = p

    # evolution equations
    #####################

	# qif population with short-term adaptation
	du[1:N] .= (v.^2 .+ η)./ τ .+ W*ls
    du[N+1:2*N] .= .- a ./ τ_d

	# lorenz system
	du[2*N+1] = lp[1] * (ls[2] - ls[1])
	du[2*N+2] = ls[1] * (lp[2] - ls[3]) - ls[2]
	du[2*N+3] = ls[1]*ls[2] - lp[3]*ls[3]

end

# definition of spike detection
spike_detection(out,u,t,integrator) = out[:] .= integrator.p[4] .- u[1:N]

# definition of the spike event handling
function spiking_mechanism(integrator, idx)

	# extract parameters and state variables
	C = integrator.p[3]
	v_th = integrator.p[4]
	α = integrator.p[6]
	J = integrator.p[9]
	v = integrator.u[1:N]
	a = integrator.u[N+1:2*N]

	# calculate network input
	s = C[:, idx];
    IJ = s .* J .* (1.0 .- a)

	# add network input to membrane potential and adaptation variable of neurons
	v[:] .+= IJ[:]
	v[idx] = -v_th
	a .+= α .* s

	# alter state variables
	integrator.u[1:N] .= v[:]
	integrator.u[N+1:2*N] .= a[:]

end

# definition of the callbacks
cb_spike = VectorContinuousCallback(spike_detection, spiking_mechanism, N, rootfind=false)

# initial condition and parameters
u0 = zeros(2*N+3,)

# connectivity matrix
rng = MersenneTwister(1234)
C = rand(rng, [0, 1], (N, N))
C_sorted = sort(C[:])
C = (C .<= C_sorted[convert(UInt32, N^2*prob)]) ./ (N*prob)

# background excitability
distr = (tan((π/2) * (2*n-N-1)/(N+1)) for n in range(1, stop=N))
η = η_b .+ Δ .* distr

# input matrix
W = rand(rng, [0, 1], (N, 3))
W_sorted = sort(W[:])
W = (W .<= W_sorted[convert(UInt32, N*3*prob_in)]) .* J_in

# initial parameters
p = [τ, η, C, v_th, τ_d, α, p_lorenz, W, J]
target_vars = range(1, stop=N)

# model setup and numerical solution
model = ODEProblem(qif_exp_lorenz, u0, tspan, p)
solution = solve(model, Tsit5(), save_idxs=target_vars, callback=cb_spike, force_dtmin=true, dense=false, dtmin=1e-12, reltol=1e-10, abstol=1e-10, saveat=dts, maxiters=1e7, progress=true)

# plotting
v_mean = mean(solution, dims=1)
plot(solution.t, v_mean', ylims=(-100.0, 100.0))

# saveing
# cutoff = 1000
# @save "qif_exp_lorenz_oscillation_small.jld2" []
# save("qif_exp_lorenz_oscillation_small.jld2", Dict("data"=>solution.u[cutoff:end]))
