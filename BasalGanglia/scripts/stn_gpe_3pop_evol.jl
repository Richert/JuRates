using Distributions, Random, Statistics, FileIO, JLD2, Distributed, DifferentialEquations, Plots, BlackBoxOptim

# definition of fixed parameters
τ_e = 15.0
τ_p = 13.0
τ_a = 22.0
τ_s = 8.0
τ_ampa_r = 0.8
τ_ampa_d = 3.7
τ_gabaa_r = 0.5
τ_gabaa_d = 5.0
k_xe = 2.4
k_xp = 3.0

# definition of the equations of motion
function stn_gpe(du, u, p, t)

    # extract state vars and params
	###############################

    r_e, v_e, r_p, v_p, r_a, v_a, r_s = u[1:7]
	I_e, y_e, E_p, x_p, I_p, y_p, E_a, x_a, I_a, y_a = u[8:17]
	r_xe, r_xp = u[[23, 26]]

    η_e_tmp, η_p_tmp, η_a_tmp, η_s, Δ_e, Δ_p, Δ_a, k_pp_tmp, k_pe_tmp, k_ps_tmp, k_ap_tmp, k_ae_tmp, k_as_tmp, k_ep_tmp = p

	# re-scale parameters
	#####################

	η_e = η_e_tmp * 10
	η_e = η_e_tmp * 10
	η_e = η_e_tmp * 10

	k_pp = k_pp_tmp * 10
	k_ep = k_ep_tmp * 10

	k_pe = k_pp*k_pe_tmp
	k_ps = k_pp*k_ps_tmp
	k_ap = k_pp*k_ap_tmp

	k_ae = k_pe*k_ae_tmp
	k_as = k_ps*k_as_tmp

    # populations
    #############

    # STN
    du[1] = (Δ_e/(π*τ_e) + 2.0*r_e*v_e) / τ_e
    du[2] = (v_e^2 + η_e - I_e*τ_e - (τ_e*π*r_e)^2) / τ_e

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
	du[8] = y_e
	du[9] = (k_ep*r_xp - 2*y_e*(τ_gabaa_r+τ_gabaa_d) - I_e)/(2*τ_gabaa_r*τ_gabaa_d)

	# at GPe-p
	du[10] = x_p
	du[11] = (k_pe*r_xe - x_p*(τ_ampa_r+τ_ampa_d) - E_p)/(τ_ampa_r*τ_ampa_d)
	du[12] = y_p
	du[13] = (k_pp*r_xp + k_ps*r_s - y_p*(τ_gabaa_r+τ_gabaa_d) - I_p)/(τ_gabaa_r*τ_gabaa_d)

	# at GPe-a
	du[14] = x_a
	du[15] = (k_ae*r_xe - x_a*(τ_ampa_r+τ_ampa_d) - E_a)/(τ_ampa_r*τ_ampa_d)
	du[16] = y_a
	du[17] = (k_ap*r_xp + k_as*r_s - y_a*(τ_gabaa_r+τ_gabaa_d) - I_a)/(τ_gabaa_r*τ_gabaa_d)

    # axonal propagation
    ####################

    # STN axons
	du[18] = k_xe * (r_e - u[18])
	du[19] = k_xe * (u[18] - u[19])
	du[20] = k_xe * (u[19] - u[20])
	du[21] = k_xe * (u[20] - u[21])
	du[22] = k_xe * (u[21] - u[22])
	du[23] = k_xe * (u[22] - u[23])

	# GPe-p axons
	du[24] = k_xp * (r_p - u[24])
	du[25] = k_xp * (u[24] - u[25])
	du[26] = k_xp * (u[25] - u[26])

end

# initial condition and parameters
N = 26
u0 = zeros(N,)
tspan = [0., 2000.]
dts = 0.1
cutoff = Int32(1000/dts)

# initial parameter values: no beta
rng = MersenneTwister(1234)

η_e = 0.0 + randn(rng, Float32)*0.5
η_p = 0.0 + randn(rng, Float32)*0.5
η_a = 0.0 + randn(rng, Float32)*0.5
η_s = 0.002

Δ_e = 0.3 + randn(rng, Float32)*0.05
Δ_p = 2.0 + randn(rng, Float32)*0.1
Δ_a = 0.5 + randn(rng, Float32)*0.05

k_pp = 3.0 + randn(rng, Float32)*0.5
k_pe = 1.5 + randn(rng, Float32)*0.2
k_ps = 5.0 + randn(rng, Float32)*0.5

k_ap = 2.0 + randn(rng, Float32)*0.2
k_ae = 0.25 + randn(rng, Float32)*0.02
k_as = 0.15 + randn(rng, Float32)*0.02

k_ep = 2.0 + randn(rng, Float32)*0.3

# initial parameters
p0 = [η_e, η_p, η_a, η_s, Δ_e, Δ_p, Δ_a, k_pp, k_pe, k_ps, k_ap, k_ae, k_as, k_ep]

# lower bounds: no beta
p_lower = [
			-3.0, # η_e
			-3.0, # η_p
			-3.0, # η_a
			0.0019, # η_s
			0.1, # Δ_e
			1.0, # Δ_p
			0.1, # Δ_a
			0.5, # k_pp
			0.8, # k_pe
			3.0, # k_ps
			1.0, # k_ap
			0.1, # k_ae
			0.05, # k_as
			0.2, # k_ep
		   ]

# upper bounds: no beta
p_upper = [
			3.0, # η_e
			3.0, # η_p
			3.0, # η_a
			0.0021, # η_s
			0.8, # Δ_e
			3.0, # Δ_p
			1.0, # Δ_a
			5.0, # k_pp
			3.0, # k_pe
			12.0, # k_ps
			3.0, # k_ap
			0.5, # k_ae
			0.2, # k_as
			5.0, # k_ep
		   ]

# firing rate targets
targets=[
        [5, 26, 4],  # healthy control
        [19, 1, 22],  # STR excitation
        [1, 10, 10],  # STN inhibition
        #[16, 76, 0],  # STN excitation
        #[missing, missing, 36]  # STN and STR excitation
]
target_vars = [1, 3, 5]

# sweep conditions
conditions = [
	([], []),
	([4], [0.03]),
	([1], [-10.0]),
	#([1], []),
	#([7], [0.2])
]

# oscillation behavior targets
freq_targets = [0.0, missing, 0.0]

# model definition
stn_gpe_prob = ODEProblem(stn_gpe, u0, tspan, p0)

# model simulation over conditions and calculation of loss
function stn_gpe_loss(p)

	# calculate new
	loss = []
	for i=1:length(targets)

		# apply condition
		indices, k_scales = conditions[i]
		p_tmp = deepcopy(p)
		for (idx, k) in zip(indices, k_scales)
			p_tmp[idx] = k
		end

		# run simulation
		sol = Array(solve(remake(stn_gpe_prob, p=p_tmp), Tsit5(), saveat=dts, reltol=1e-8, abstol=1e-8)) .* 1e3

		# calculate loss
		target = targets[i]
		freq_target = freq_targets[i]

		diff1 = sum(((mean(sol[j, cutoff:end])-t)/t)^2 for (j,t) in zip(target_vars, target) if ! ismissing(t))
	    diff2 = ismissing(freq_target) ? 1/var(sol[3,cutoff:end]) : var(sol[3, cutoff:end])
		r_max = maximum(maximum(sol[target_vars, cutoff:end]))
		diff3 = r_max > 500.0 ? r_max - 500.0 : 0.0
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
    sol = Array(solve(remake(stn_gpe_prob, p=p), Tsit5(), saveat=dts, reltol=1e-8, abstol=1e-8))
	display(plot(sol[target_vars, cutoff:end]' .* 1e3))
 return push!(fitness_progress_history, (BlackBoxOptim.num_func_evals(oc), best_fitness(oc)))
end

# choose optimization algorithm
method = :dxnes

# start optimization
opt = bbsetup(stn_gpe_loss; Method=method, Parameters=p0, SearchRange=(collect(zip(p_lower,p_upper))), NumDimensions=length(p0), MaxSteps=100, workers=workers(), TargetFitness=0.0, PopulationSize=5000, CallbackFunction=cb, CallbackInterval=1.0)
el = @elapsed res = bboptimize(opt)
t = round(el, digits=3)

# receive optimization results
p = best_candidate(res)
f = best_fitness(res)
display(p)
η_e, η_p, η_a, η_s, Δ_e, Δ_p, Δ_a, k_pp, k_pe, k_ps, k_ap, k_ae, k_as, k_ep = p

sol = solve(remake(stn_gpe_prob, p=p), Tsit5(), saveat=dts, reltol=1e-8, abstol=1e-8)
display(plot(sol[target_vars, cutoff:]' .* 1e3))

# store best parameter set
jname = "test_params"#ARGS[1]
jid = 0#ARGS[2]
@save "BasalGanglia/results/$jname" * "_$jid" * "_params.jdl" p
@save "BasalGanglia/results/$jname" * "_$jid" * "_fitness.jdl" f
