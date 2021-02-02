using DifferentialEquations,Plots,Statistics,JLD2,DSP

# definition of fixed parameters
τ = 1.0
τ_d = 10.0
Δ = 2.0
J = 15.0*√Δ
η_b = -5.15
N = 4000
m = 1
prob = 0.1
v_th = 1000.0
α = 0.05

# simulation parameters
tspan = [0., 40.0]
dts = 0.002

# definition of cortical input
# stim_t = 50.0
# stim_v = 8.0
# stim_off = 8.0
# stim_cut = 10.0
# ctx_t = truncated(Normal(stim_t, stim_v), stim_t - stim_cut, stim_t + stim_cut)
# str_t = truncated(Normal(stim_t+stim_off, stim_v), stim_t+stim_off-stim_cut, stim_t+stim_off+stim_cut)

# definition of the equations of motion
function qif_exp(du, u, p, t)

    # extract state vars and params
	###############################

    v, a = u[1:N], u[N+1:2*N]
    τ, η, C, v_th, τ_d, α, J = p

    # evolution equations
    #####################

	du[1:N] .= (v.^2 .+ η)./ τ
    du[N+1:2*N] .= .- a ./ τ_d

end

# definition of spike detection
spike_detection(out,u,t,integrator) = out[:] .= integrator.p[4] .- u[1:N]

# definition of the spike event handling
function spiking_mechanism(integrator, idx)

	# extract parameters and state variables
	C = integrator.p[3]
	v_th = integrator.p[4]
	α = integrator.p[6]
	J = integrator.p[7]
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
u0 = zeros(2*N,)

# connectivity matrix
C = rand([0, 1], (N, N))
C_sorted = sort(C[:])
C = (C .<= C_sorted[convert(UInt32, N^2*prob)]) ./ (N*prob)

# background excitability
distr = (tan((π/2) * (2*n-N-1)/(N+1)) for n in range(1, stop=N))
η = η_b .+ Δ .* distr

# initial parameters
p = [τ, η, C, v_th, τ_d, α, J]
target_vars = range(1, stop=N, step=m)

# model setup and numerical solution
model = ODEProblem(qif_exp, u0, tspan, p)
solution = solve(model, Tsit5(), save_idxs=target_vars, callback=cb_spike, force_dtmin=true, dense=false, dtmin=1e-12, reltol=1e-10, abstol=1e-10, saveat=dts, maxiters=1e7, progress=true)

# calculate firing rate
n = convert(Int32, N/m)
v = zeros((n, length(solution.u)))
for i=1:length(solution.u)
    v[:, i] .= solution.u[i][1:n]
end
fr = 1.0 .* ((v .+ v_th).^2  .< 0.1)
fr_mean = mean(fr, dims=1)
fr_mean_filt = filtfilt(gaussian(1000, 0.5), fr_mean[1, :])

# plotting
plot(solution.t, fr_mean_filt)
