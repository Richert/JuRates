using Distributions, Random, Statistics, FileIO, JLD2, Distributed, DifferentialEquations, Plots, BlackBoxOptim

# definition of the motion equations
τ_e = 13.0
τ_p = 25.0
τ_a = 20.0

Δ_e = 0.05*τ_e^2
Δ_p = 0.6*τ_p^2
Δ_a = 0.3*τ_a^2

function stn_gpe(du, u, p, t)

    # extract state vars and params
	###############################
    r_e, v_e, r_p, v_p, r_a, v_a = u[1:6]
	r_xe, r_ep, r_xp, r_xa, r_ee = u[[22, 38, 42, 46, 50]]
    η_e, η_p, η_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as = p

	# set/adjust parameters
	#######################

	k_pe_d = 4
	k_ep_d = 5
	k_p_d = 4
	k_a_d = 4
	k_ee_d = 4

    # populations
    #############

    # STN
    du[1] = (Δ_e/(π*τ_e) + 2.0*r_e*v_e) / τ_e
    du[2] = (v_e^2 + η_e + (k_ee*r_ee - k_ep*r_ep)*τ_e - (τ_e*π*r_e)^2) / τ_e

    # GPe-p
    du[3] = (Δ_p/(π*τ_p) + 2.0*r_p*v_p) / τ_p
    du[4] = (v_p^2 + η_p + (k_pe*r_xe - k_pp*r_xp - k_pa*r_xa - k_ps*0.002)*τ_p - (τ_p*π*r_p)^2) / τ_e

    # GPe-a
    du[5] = (Δ_a/(π*τ_a) + 2.0*r_a*v_a) / τ_a
    du[6] = (v_a^2 + η_a + (k_ae*r_xe - k_ap*r_xp - k_aa*r_xa - k_as*0.002)*τ_a - (τ_a*π*r_a)^2) / τ_a

    # axonal propagation
    ####################

    # STN to GPe-p
	du[7] = k_pe_d * (r_e - u[7])
	du[8] = k_pe_d * (u[7] - u[8])
	du[9] = k_pe_d * (u[8] - u[9])
	du[10] = k_pe_d * (u[9] - u[10])
	du[11] = k_pe_d * (u[10] - u[11])
	du[12] = k_pe_d * (u[11] - u[12])
	du[13] = k_pe_d * (u[12] - u[13])
	du[14] = k_pe_d * (u[13] - u[14])
	du[15] = k_pe_d * (u[14] - u[15])
	du[16] = k_pe_d * (u[15] - u[16])
	du[17] = k_pe_d * (u[16] - u[17])
	du[18] = k_pe_d * (u[17] - u[18])
	du[19] = k_pe_d * (u[18] - u[19])
	du[20] = k_pe_d * (u[19] - u[20])
	du[21] = k_pe_d * (u[20] - u[21])
	du[22] = k_pe_d * (u[21] - u[22])

	# GPe-p to STN
	du[23] = k_ep_d * (r_p - u[23])
	du[24] = k_ep_d * (u[23] - u[24])
	du[25] = k_ep_d * (u[24] - u[25])
	du[26] = k_ep_d * (u[25] - u[26])
	du[27] = k_ep_d * (u[26] - u[27])
	du[28] = k_ep_d * (u[27] - u[28])
	du[29] = k_ep_d * (u[28] - u[29])
	du[30] = k_ep_d * (u[29] - u[30])
	du[31] = k_ep_d * (u[30] - u[31])
	du[32] = k_ep_d * (u[31] - u[32])
	du[33] = k_ep_d * (u[32] - u[33])
	du[34] = k_ep_d * (u[33] - u[34])
	du[35] = k_ep_d * (u[34] - u[35])
	du[36] = k_ep_d * (u[35] - u[36])
	du[37] = k_ep_d * (u[36] - u[37])
	du[38] = k_ep_d * (u[37] - u[38])

	# Gpe-p to both GPes
	du[39] = k_p_d * (r_p - u[39])
	du[40] = k_p_d * (u[39] - u[40])
	du[41] = k_p_d * (u[40] - u[41])
	du[42] = k_p_d * (u[41] - u[42])

	# ! Gpe-a to both GPes
	du[43] = k_a_d * (r_a - u[43])
	du[44] = k_a_d * (u[43] - u[44])
	du[45] = k_a_d * (u[44] - u[45])
	du[46] = k_a_d * (u[45] - u[46])

    # ! STN to STN
	du[47] = k_ee_d * (r_e - u[47])
	du[48] = k_ee_d * (u[47] - u[48])
	du[49] = k_ee_d * (u[48] - u[49])
	du[50] = k_ee_d * (u[49] - u[50])

end

# initial condition and parameters
N = 50
N1 = 30
u0 = zeros(N,)
tspan = [0., 50.]

#rng = MersenneTwister(1234)

η_e = 1*Δ_e
η_p = -0.5*Δ_p
η_a = -2.0*Δ_a

k_ee = 2.0*√Δ_e
k_pe = 200.0*√Δ_p
k_ae = 100.0*√Δ_a
k_pp = 7.0*√Δ_p
k_ep = 50.0*√Δ_e
k_ap = 20.0*√Δ_a
k_pa = 20.0*√Δ_p
k_aa = 5.0*√Δ_a
k_ps = 100.0*√Δ_p
k_as = 100.0*√Δ_a

# initial parameters
p0 = [η_e, η_p, η_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as]
#@load "BasalGanglia/results/stn_gpe_params.jld" p

# lower bounds
p_lower = [-10*Δ_e, -10*Δ_p, -20*Δ_a, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# upper bounds
p_upper = [10*Δ_e, 10*Δ_p, 20*Δ_a, 10*√Δ_e, 400*√Δ_p, 200*√Δ_a, 400*√Δ_e, 20*√Δ_p, 100*√Δ_a, 100*√Δ_p, 20*√Δ_a, 200*√Δ_p, 400*√Δ_a]

# firing rate targets
targets=[[20, 60, 30],  # healthy control
         [missing, 30, missing],  # ampa blockade in GPe
         [missing, 70, missing],  # ampa and gabaa blockade in GPe
         [missing, 100, missing],  # GABAA blockade in GPe
         [40, 100, missing]  # GABAA blockade in STN
        ]
target_vars = [1, 3, 5]

# sweep conditions
conditions = [
	([], []),
	([5, 6], [0.2, 0.2]),
	([5, 6, 8, 9, 10, 11, 12, 13], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
	([9, 10, 11, 12, 13], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
	([7], [0.2])
]

# oscillation behavior targets
freq_targets = [0.0, 0.0, missing, 0.0, missing]

# model sweep function
function stn_gpe_conditions(prob,i,repeat)
	indices, k_scales = conditions[i]
	p = [prob.p[j]*k for (j, k) in zip(indices, k_scales)]
	remake(prob, p=p)
	prob
end

# model definition
stn_gpe_prob = ODEProblem(stn_gpe, u0, tspan, p0)
stn_gpe_sweep = EnsembleProblem(stn_gpe_prob, prob_func=stn_gpe_conditions)

# model simulation over conditions and calculation of loss
function stn_gpe_loss(p)

	diff1, diff2, diff3 = 0, 0, 0
	for i=1:length(targets)

	    # run simulation
	    sol = solve(remake(stn_gpe_prob, p=p), DP5(), saveat=0.1, reltol=1e-4, abstol=1e-6)

		# calculate loss
		sol_tmp = sol[target_vars, :] .* 1e3
		target = targets[i]
		freq_target = freq_targets[i]

		diff1 += sum((s-t)^2/t for (s,t) in zip(mean(sol_tmp[:, 400:end],dims=2), target) if ! ismissing(t))
	    diff2 += ismissing(freq_target) ? 1/var(sol_tmp[2, 50:end]) : var(sol_tmp[2, 50:end])
		r_max = maximum(maximum(abs.(sol_tmp[:, 200:end])))
		diff3 += r_max^2 > 1000.0 ? r_max^2 : 0.0
	end

    return diff1 + diff2^2 + diff3
end

# callback function
fitness_progress_history = Array{Tuple{Int, Float64},1}()
cb = function (oc) #callback function to observe training
	p = best_candidate(oc)
    sol = solve(remake(stn_gpe_prob, p=p), DP5(), saveat=0.1, reltol=1e-4, abstol=1e-6) .* 1e3
	display(plot(sol[target_vars, :]'))
  return push!(fitness_progress_history, (BlackBoxOptim.num_func_evals(oc), best_fitness(oc)))
end

# choose optimization algorithm
method = :dxnes

# start optimization
opt = bbsetup(stn_gpe_loss; Method=method, SearchRange=(collect(zip(p_lower,p_upper))), NumDimensions=length(p0), MaxSteps=10000, workers=workers(), TargetFitness=1.0, PopulationSize=5000)
#, CallbackFunction = cb, CallbackInterval=0.0)
el = @elapsed res = bboptimize(opt)
t = round(el, digits=3)

# receive optimization results
p_new = best_candidate(res)
display(p_new)
η_e, η_p, η_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as = p_new

# store best parameter set
jname = ARGS[1]
jid = ARGS[2]
@save "results/$jname" * "_$jid" * "_params.jdl" p_new
