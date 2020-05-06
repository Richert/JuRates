using Distributions, Random, Statistics, FileIO, JLD2, Distributed, DifferentialEquations, Plots, BlackBoxOptim

# definition of fixed parameters
τ_e = 13.0
τ_p = 25.0
τ_a = 20.0
τ_e_ampa_r = 0.8
τ_e_ampa_d = 3.7
τ_p_ampa_r = 0.8
τ_p_ampa_d = 3.7
τ_a_ampa_r = 0.8
τ_a_ampa_d = 3.7
τ_e_gabaa_r = 0.8
τ_e_gabaa_d = 10.0
τ_p_gabaa_r = 0.5
τ_p_gabaa_d = 5.0
τ_a_gabaa_r = 0.5
τ_a_gabaa_d = 5.0

function stn_gpe(du, u, p, t)

    # extract state vars and params
	###############################
    r_e, v_e, r_p, v_p, r_a, v_a, r_s = u[1:7]
	r_xe, r_ep, r_xp, r_xa, r_ee = u[[27, 31, 33, 35, 37]]
	E_e, x_e, I_e, y_e, E_p, x_p, I_p, y_p, E_a, x_a, I_a, y_a = u[8:19]
    η_e, η_p, η_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as, Δ_e, Δ_p, Δ_a = p

	# set/adjust parameters
	#######################

	k_ee_d = 1.14  # order = 2
	k_pe_d = 2.67  # order = 8
	k_ep_d = 2.0  # order = 4
	k_p_d = 1.33  # order = 2
	k_a_d = 1.33  # order = 2
	η_s = 0.002
	τ_s = 1.0

	Δ_e = Δ_e*τ_e^2
	Δ_p = Δ_p*τ_p^2
	Δ_a = Δ_a*τ_a^2

	η_e = η_e*Δ_e
	η_p = η_p*Δ_p
	η_a = η_a*Δ_a

	k_ee = k_ee*√Δ_e
	k_pe = k_pe*√Δ_p
	k_ae = k_ae*√Δ_a
	k_ep = k_ep*√Δ_e
	k_pp = k_pp*√Δ_p
	k_ap = k_ap*√Δ_a
	k_pa = k_pa*√Δ_p
	k_aa = k_aa*√Δ_a
	k_ps = k_ps*√Δ_p
	k_as = k_as*√Δ_a

    # populations
    #############

    # STN
    du[1] = (Δ_e/(π*τ_e) + 2.0*r_e*v_e) / τ_e
    du[2] = (v_e^2 + η_e + (E_e - I_e)*τ_e - (τ_e*π*r_e)^2) / τ_e

    # GPe-p
    du[3] = (Δ_p/(π*τ_p) + 2.0*r_p*v_p) / τ_p
    du[4] = (v_p^2 + η_p + (E_p - I_p)*τ_p - (τ_p*π*r_p)^2) / τ_p

    # GPe-a
    du[5] = (Δ_a/(π*τ_a) + 2.0*r_a*v_a) / τ_a
    du[6] = (v_a^2 + η_a + (E_a - I_a)*τ_a - (τ_a*π*r_a)^2) / τ_a

	# dummy STR
	du[7] = (η_s - r_s) / τ_s

	# synapse dynamics
    ##################

	# at STN
	du[8] = x_e
	du[9] = (k_ee*r_ee - x_e*(τ_e_ampa_r+τ_e_ampa_d) - E_e)/(τ_e_ampa_r*τ_e_ampa_d)
	du[10] = y_e
	du[11] = (k_ep*r_ep - y_e*(τ_e_gabaa_r+τ_e_gabaa_d) - I_e)/(τ_e_gabaa_r*τ_e_gabaa_d)

	# at GPe-p
	du[12] = x_p
	du[13] = (k_pe*r_xe - x_p*(τ_p_ampa_r+τ_p_ampa_d) - E_p)/(τ_p_ampa_r*τ_p_ampa_d)
	du[14] = y_p
	du[15] = (k_pp*r_xp + k_pa*r_xa + k_ps*r_s - y_p*(τ_p_gabaa_r+τ_p_gabaa_d) - I_p)/(τ_p_gabaa_r*τ_p_gabaa_d)

	# at GPe-a
	du[16] = x_a
	du[17] = (k_ae*r_xe - x_a*(τ_a_ampa_r+τ_a_ampa_d) - E_a)/(τ_a_ampa_r*τ_a_ampa_d)
	du[18] = y_a
	du[19] = (k_ap*r_xp + k_aa*r_xa + k_as*r_s - y_a*(τ_a_gabaa_r+τ_a_gabaa_d) - I_a)/(τ_a_gabaa_r*τ_a_gabaa_d)

    # axonal propagation
    ####################

    # STN to both GPe
	du[20] = k_pe_d * (r_e - u[20])
	du[21] = k_pe_d * (u[20] - u[21])
	du[22] = k_pe_d * (u[21] - u[22])
	du[23] = k_pe_d * (u[22] - u[23])
	du[24] = k_pe_d * (u[23] - u[24])
	du[25] = k_pe_d * (u[24] - u[25])
	du[26] = k_pe_d * (u[25] - u[26])
	du[27] = k_pe_d * (u[26] - u[27])

	# GPe-p to STN
	du[28] = k_ep_d * (r_p - u[28])
	du[29] = k_ep_d * (u[28] - u[29])
	du[30] = k_ep_d * (u[29] - u[30])
	du[31] = k_ep_d * (u[30] - u[31])

	# Gpe-p to both GPes
	du[32] = k_p_d * (r_p - u[32])
	du[33] = k_p_d * (u[32] - u[33])

	# Gpe-a to both GPes
	du[34] = k_a_d * (r_a - u[34])
	du[35] = k_a_d * (u[34] - u[35])

	# STN to STN
	du[36] = k_ee_d * (r_e - u[36])
	du[37] = k_ee_d * (u[36] - u[37])

end

# initial condition and parameters
N = 37
u0 = zeros(N,)
tspan = [0., 200.]

#rng = MersenneTwister(1234)
Δ_e = 0.1
Δ_p = 0.3
Δ_a = 0.2

η_e = 0.0
η_p = 0.0
η_a = 0.0

k_ee = 3.0
k_pe = 80.0
k_ae = 30.0
k_ep = 30.0
k_pp = 6.0
k_ap = 30.0
k_pa = 60.0
k_aa = 4.0
k_ps = 80.0
k_as = 160.0

# initial parameters
p = [η_e, η_p, η_a,
	k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as,
	Δ_e, Δ_p, Δ_a]
#@load "BasalGanglia/results/stn_gpe_params.jld" p

# lower bounds
p_lower = [-4.0, # η_e
		   -4.0, # η_p
		   -6.0, # η_a
		   0.0, # k_ee
		   1.0, # k_pe
		   1.0, # k_ae
		   1.0, # k_ep
		   0.0, # k_pp
		   1.0, # k_ap
		   1.0, # k_pa
		   0.0, # k_aa
		   1.0, # k_ps
		   3.0, # k_as
		   0.01, # Δ_e
		   0.1, # Δ_p
		   0.02, # Δ_a
		   ]

# upper bounds
p_upper = [4.0, # η_e
		   4.0, # η_p
		   2.0, # η_a
		   10.0, # k_ee
		   200.0, # k_pe
		   200.0, # k_ae
		   200.0, # k_ep
		   50.0, # k_pp
		   200.0, # k_ap
		   200.0, # k_pa
		   50.0, # k_aa
		   200.0, # k_ps
		   400.0, # k_as
		   0.1, # Δ_e
		   0.4, # Δ_p
		   0.2, # Δ_a
		   ]

# firing rate targets
targets=[[19, 62, 31],  # healthy control
         [missing, 35, missing],  # ampa blockade in GPe
         [missing, 76, missing],  # ampa and gabaa blockade in GPe
         [missing, 135, missing],  # GABAA blockade in GPe
         [35, 124, missing]  # GABAA blockade in STN
        ]
target_vars = [1, 3, 5]

# sweep conditions
conditions = [
	([], []),
	([5, 6], [0.2, 0.2]),
	([5, 6, 8, 9, 10, 11, 12, 13], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
	([8, 9, 10, 11, 12, 13], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
	([7], [0.2])
]

# oscillation behavior targets
freq_targets = [0.0, 0.0, 0.0, 0.0, missing]

# model definition
stn_gpe_prob = ODEProblem(stn_gpe, u0, tspan, p)

# model simulation over conditions and calculation of loss
function stn_gpe_loss(p)

	# calculate new
	loss = []
	for i=1:length(targets)

		# apply condition
		indices, k_scales = conditions[i]
		p_tmp = deepcopy(p)
		for (idx, k) in zip(indices, k_scales)
			p_tmp[idx] = p_tmp[idx] * k
		end

		# run simulation
		sol = Array(solve(remake(stn_gpe_prob, p=p_tmp), DP5(), saveat=0.1, reltol=1e-4, abstol=1e-6)) .* 1e3

		# calculate loss
		target = targets[i]
		freq_target = freq_targets[i]

		diff1 = sum(((mean(sol[j, 1000:end])-t)^2)/t for (j,t) in zip(target_vars, target) if ! ismissing(t))
	    diff2 = ismissing(freq_target) ? 1/var(sol[2,200:end]) : var(sol[2, 1000:end])
		r_max = maximum(maximum(sol[target_vars, :]))
		diff3 = r_max > 800.0 ? r_max - 800.0 : 0.0
		push!(loss, diff1 + √diff2 + diff3)
	end

	# save new parameterization
	remake(stn_gpe_prob, p=p)

	# return loss
    sum(loss)
end

# callback function
fitness_progress_history = Array{Tuple{Int, Float64},1}()
cb = function (oc) #callback function to observe training
	p = best_candidate(oc)
    sol = Array(solve(remake(stn_gpe_prob, p=p), DP5(), reltol=1e-4, abstol=1e-6, save_idxs=target_vars, saveat=0.1))
	#display(plot(sol' .* 1e3))
  return push!(fitness_progress_history, (BlackBoxOptim.num_func_evals(oc), best_fitness(oc)))
end

# choose optimization algorithm
method = :dxnes

# start optimization
opt = bbsetup(stn_gpe_loss; Method=method, Parameters=p0, SearchRange=(collect(zip(p_lower,p_upper))), NumDimensions=length(p0), MaxSteps=4000, workers=workers(), TargetFitness=0.0, PopulationSize=10000)
el = @elapsed res = bboptimize(opt)
t = round(el, digits=3)

# receive optimization results
p = best_candidate(res)
f = best_fitness(res)
display(p)
η_e, η_p, η_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as, Δ_e, Δ_p, Δ_a = p

#sol = solve(remake(stn_gpe_prob, p=p), DP5(), saveat=0.1, reltol=1e-4, abstol=1e-6)
#display(plot(sol[target_vars, :]' .* 1e3))

# store best parameter set
jname = ARGS[1]
jid = ARGS[2]
@save "results/$jname" * "_$jid" * "_params.jdl" p
@save "results/$jname" * "_$jid" * "_fitness.jdl" f
