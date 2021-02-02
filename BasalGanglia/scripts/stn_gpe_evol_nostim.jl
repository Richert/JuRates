using Distributions, Random, Statistics, FileIO, JLD2, Distributed, DifferentialEquations, BlackBoxOptim, DSP

function stn_gpe(du, u, p, t)

    # extract state vars and params
	###############################
    r_e, v_e, r_p, v_p, r_s = u[1:5]
	r_pe, r_ep, r_pp, r_ee = u[[21, 25, 27, 29]]
	E_e, x_e, I_e, y_e, E_p, x_p, I_p, y_p = u[6:13]
    τ_e, τ_p, τ_ampa_r, τ_ampa_d, τ_gabaa_r, τ_gabaa_d, τ_gabaa_stn, η, Δ, k, η_e, η_p, k_pe, k_ep, k_pp = p

	# set/adjust parameters
	#######################

	η_e = η_e*η
	η_p = η_p*η

	Δ_e = 0.3*Δ
	Δ_p = 0.9*Δ

	k_ee = 0.8*k
	k_pe = k_pe*k
	k_ep = k_ep*k
	k_pp = k_pp*k
	k_ps = 20.0*k

	k_ee_d = 1.14  # order = 2
	k_pe_d = 2.67  # order = 8
	k_ep_d = 2.0  # order = 4
	k_p_d = 1.33  # order = 2
	η_s = 0.002
	τ_s = 1.0

    # populations
    #############

    # STN
    du[1] = (Δ_e/(π*τ_e) + 2.0*r_e*v_e) / τ_e
    du[2] = (v_e^2 + η_e + (E_e - I_e)*τ_e - (τ_e*π*r_e)^2) / τ_e

    # GPe-p
    du[3] = (Δ_p/(π*τ_p) + 2.0*r_p*v_p) / τ_p
    du[4] = (v_p^2 + η_p + (E_p - I_p)*τ_p - (τ_p*π*r_p)^2) / τ_p

	# dummy STR
	du[5] = (η_s - r_s) / τ_s

	# synapse dynamics
    ##################

	# at STN
	du[6] = x_e
	du[7] = (k_ee*r_ee - x_e*(τ_ampa_r+τ_ampa_d) - E_e)/(τ_ampa_r*τ_ampa_d)
	du[8] = y_e
	du[9] = (k_ep*r_ep - τ_gabaa_stn*y_e*(τ_gabaa_r+τ_gabaa_d) - I_e)/(τ_gabaa_r*τ_gabaa_d*τ_gabaa_stn^2)

	# at GPe-p
	du[10] = x_p
	du[11] = (k_pe*r_pe - x_p*(τ_ampa_r+τ_ampa_d) - E_p)/(τ_ampa_r*τ_ampa_d)
	du[12] = y_p
	du[13] = (k_pp*r_pp + k_ps*r_s - y_p*(τ_gabaa_r+τ_gabaa_d) - I_p)/(τ_gabaa_r*τ_gabaa_d)

    # axonal propagation
    ####################

    # STN to GPe-p
	du[14] = k_pe_d * (r_e - u[14])
	du[15] = k_pe_d * (u[14] - u[15])
	du[16] = k_pe_d * (u[15] - u[16])
	du[17] = k_pe_d * (u[16] - u[17])
	du[18] = k_pe_d * (u[17] - u[18])
	du[19] = k_pe_d * (u[18] - u[19])
	du[20] = k_pe_d * (u[19] - u[20])
	du[21] = k_pe_d * (u[20] - u[21])

	# GPe-p to STN
	du[22] = k_ep_d * (r_p - u[22])
	du[23] = k_ep_d * (u[22] - u[23])
	du[24] = k_ep_d * (u[23] - u[24])
	du[25] = k_ep_d * (u[24] - u[25])

	# Gpe-p to Gpe-p
	du[26] = k_p_d * (r_p - u[26])
	du[27] = k_p_d * (u[26] - u[27])

	# STN to STN
	du[28] = k_ee_d * (r_e - u[28])
	du[29] = k_ee_d * (u[28] - u[29])

end

# initial condition and parameters
N = 29
u0 = zeros(N,)
tspan = [0., 11000.]
dts = 0.1
cutoff = Int32(1000/dts)

#rng = MersenneTwister(1234)
τ_e = 13.0
τ_p = 25.0
τ_ampa_r = 0.8
τ_ampa_d = 3.7
τ_gabaa_r = 0.5
τ_gabaa_d = 5.0
τ_gabaa_stn = 2.0
η = 100.0
Δ = 100.0
k = 100.0
η_e = 4.0
η_p = 3.145
k_pe = 8.0
k_ep = 10.0
k_pp = 5.0

# initial parameters
p = [τ_e, τ_p, τ_ampa_r, τ_ampa_d, τ_gabaa_r, τ_gabaa_d, τ_gabaa_stn, η, Δ, k, η_e, η_p, k_pe, k_ep, k_pp]

# lower bounds
p_lower = [5.0, # τ_e
		   10.0, # τ_p
		   0.4, # τ_ampa_r
		   3.0, # τ_ampa_d
		   0.4, # τ_gabaa_r
		   4.0, # τ_gabaa_d
		   1.0, # τ_gabaa_stn
		   1.0, # η
		   1.0, # Δ
		   1.0, # k
		   -5.0, # η_e
		   -5.0, # η_p
		   3.0, # k_pe
		   5.0, # k_ep
		   1.0, # k_pp
		   ]

# upper bounds
p_upper = [25.0, # τ_e
		   50.0, # τ_p
		   2.0, # τ_ampa_r
		   10.0, # τ_ampa_d
		   2.0, # τ_gabaa_r
		   20.0, # τ_gabaa_d
		   3.0, # τ_gabaa_stn
		   1000.0, # η
		   1000.0, # Δ
		   1000.0, # k
		   10.0, # η_e
		   10.0, # η_p
		   8.0, # k_pe
		   15.0, # k_ep
		   5.0, # k_pp
		   ]

# loss function parameters
freq_target = 14.0
α = 0.04
β = 0.04
γ = 0.01

# model definition
stn_gpe_prob = ODEProblem(stn_gpe, u0, tspan, p)

# model simulation and calculation of loss
function stn_gpe_loss(p)

	# run simulation
	sol = Array(solve(remake(stn_gpe_prob, p=p), Tsit5(), saveat=dts, reltol=1e-8, abstol=1e-8)) .* 1e3

	# calculate psd profiles
	s = sol[1, cutoff:end]
	n = div(length(s), 8)
	psd_stn = welch_pgram(s, n, div(n, 2); nfft=nextfastfft(n), fs=1e3/dts, window=nothing)
	freqs_stn = Array(freq(psd_stn))
	stn_idx = 2.0 .< freqs_stn .< 200.0

	s2 = sol[3, cutoff:end]
	psd_gpe = welch_pgram(s2, n, div(n, 2); nfft=nextfastfft(n), fs=1e3/dts, window=nothing)
	freqs_gpe = Array(freq(psd_gpe))
	gpe_idx = 2.0 .< freqs_gpe .< 200.0

	# calculate loss
	stn_p = psd_stn.power[stn_idx]
	stn_f = psd_stn.freq[stn_idx]
	gpe_p = psd_gpe.power[gpe_idx]
	gpe_f = psd_gpe.freq[gpe_idx]
	max_stn = argmax(stn_p)
	max_gpe = argmax(gpe_p)
	r1 = 1/gpe_p[max_gpe]^2 + 1/stn_p[max_stn]^2
	r2 = gpe_p[max_gpe] + stn_p[max_stn]
	r3 = (40-mean(s2))^2 + (30 - mean(s))^2
	loss = (stn_f[max_stn] - freq_target)^2 + (gpe_f[max_gpe] - freq_target)^2 + α*r1 + β*r2 + γ*r3

	# save new parameterization
	remake(stn_gpe_prob, p=p)

	# return loss
    loss
end

# callback function
# fitness_progress_history = Array{Tuple{Int, Float64},1}()
# cb = function (oc) #callback function to observe training
#
# 	# simulate behavior of best candidate
# 	p = best_candidate(oc)
#     sol = Array(solve(remake(stn_gpe_prob, p=p), Tsit5(), saveat=dts, reltol=1e-8, abstol=1e-8)) .* 1e3
#
# 	# calculate PSD
# 	s = sol[3, cutoff:end]
# 	n = div(length(s), 8)
# 	psd = welch_pgram(s, n, div(n, 2); nfft=nextfastfft(n), fs=1e3/dts, window=nothing)
# 	freqs = Array(freq(psd))
# 	freq_idx = 2.0 .< freqs .< 200.0
#
# 	# plot firing rate and PSD profile
# 	p1 = plot(sol[[1, 3],cutoff:5000+cutoff]')
#
# 	p2 = plot(psd.freq[freq_idx], psd.power[freq_idx])
# 	display(plot(p1, p2, layout=(2,1)))
#
#   return push!(fitness_progress_history, (BlackBoxOptim.num_func_evals(oc), best_fitness(oc)))
# end

# choose optimization algorithm
method = :dxnes

# start optimization: add callback via CallbackFunction=cb, CallbackInterval=1.0
opt = bbsetup(stn_gpe_loss; Method=method, Parameters=p, SearchRange=(collect(zip(p_lower,p_upper))), NumDimensions=length(p), MaxSteps=500, workers=workers(), TargetFitness=0.0, PopulationSize=10000)

el = @elapsed res = bboptimize(opt)
t = round(el, digits=3)

# receive optimization results
p = best_candidate(res)
f = best_fitness(res)
display(p)
τ_e, τ_p, τ_ampa_r, τ_ampa_d, τ_gabaa_r, τ_gabaa_d, τ_gabaa_stn, η, Δ, k, η_e, η_p, k_pe, k_ep, k_pp = p

sol = Array(solve(remake(stn_gpe_prob, p=p), Tsit5(), saveat=dts, reltol=1e-8, abstol=1e-8)) .* 1e3
# display(plot(sol[[1, 3], cutoff:cutoff+5000]'))

# store best parameter set
# @save "/home/rgast/JuliaProjects/JuRates/BasalGanglia/results/stn_gpe_3pop_p3.jdl" p
# @save "/home/rgast/JuliaProjects/JuRates/BasalGanglia/results/stn_gpe_3pop_f3.jdl" f

# store best parameter set
jname = ARGS[1]
jid = ARGS[2]
@save "../../JuRates/BasalGanglia/results/$jname" * "_$jid" * "_params.jdl" p
@save "../../JuRates/BasalGanglia/results/$jname" * "_$jid" * "_fitness.jdl" f
