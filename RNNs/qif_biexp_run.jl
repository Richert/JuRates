using DifferentialEquations,Plots

# definition of fixed parameters
τ = 1.0
τ_r = 10.0
τ_d = 10.0
Δ = 2.0
J = 15.0*√Δ
η_b = -5.22
N = 1000
prob = 0.05
v_th = 1000.0
α = 0.05

# definition of cortical input
# stim_t = 50.0
# stim_v = 8.0
# stim_off = 8.0
# stim_cut = 10.0
# ctx_t = truncated(Normal(stim_t, stim_v), stim_t - stim_cut, stim_t + stim_cut)
# str_t = truncated(Normal(stim_t+stim_off, stim_v), stim_t+stim_off-stim_cut, stim_t+stim_off+stim_cut)

# definition of the equations of motion
function qif_biexp(du, u, p, t)

    # extract state vars and params
	###############################

    v, a, x = u[1:N], u[N+1:2*N], u[2*N+1:3*N]
    τ, η, C, v_th, τ_r, τ_d, α = p

    # evolution equations
    #####################

	du[1:N] .= (v.^2 .+ η)./ τ;
    du[N+1:2*N] .= x;
    du[2*N+1:3*N] .= .- ((τ_r+τ_d).*x .+ a) ./ (τ_r*τ_d);

end

# definition of spike detection
spike_detection(out,u,t,integrator) = out[:] .= integrator.p[4] .- u[1:N]

# definition of the spike event handling
function spiking_mechanism(integrator, idx)

	# extract parameters and state variables
	C = integrator.p[3]
	v_th = integrator.p[4]
	τ_r = integrator.p[5]
	α = integrator.p[7]
	v = integrator.u[1:N]
	x = integrator.u[2*N+1:3*N]

	# calculate network input
	s = C[:, idx];
    IJ = s .* (1.0 .- integrator.u[N+1:2*N])

	# add network input to membrane potential and adaptation variable of neurons
	v[:] .+= IJ[:]
	v[idx] = -v_th
	x .+= α .* s ./ τ_r

	# alter state variables
	integrator.u[1:N] .= v[:]
	integrator.u[2*N+1:3*N] .= x[:]

end

# definition of the callbacks
cb_spike = VectorContinuousCallback(spike_detection, spiking_mechanism, N, rootfind=false)

# initial condition and parameters
u0 = zeros(3*N,)
tspan = [0., 80.]

# connectivity matrix
C = rand([0, 1], (N, N))
C_sorted = sort(C[:])
C = (C .<= C_sorted[convert(UInt32, N^2*prob)]) .* J ./ (N*prob)

# background excitability
distr = (tan((π/2) * (2*n-N-1)/(N+1)) for n in range(1, stop=N))
η = η_b .+ Δ .* distr

# initial parameters
p = [τ, η, C, v_th, τ_r, τ_d, α]
target_vars = range(1, stop=N)

# model setup and numerical solution
model = ODEProblem(qif_biexp, u0, tspan, p)
solution = solve(model, Tsit5(), save_idxs=target_vars, callback=cb_spike, force_dtmin=true, dense=false, dtmin=1e-12, reltol=1e-10, abstol=1e-10)

# plotting
plot(solution.t, solution', ylims=(-v_th, v_th))
