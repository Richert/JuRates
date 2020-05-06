using Distributions, Random, Statistics, FileIO, JLD2, Distributed, DifferentialEquations, Plots, BlackBoxOptim

# definition of the motion equations
function stn_gpe(du, u, p, t)

    # extract state vars and params
	###############################
    r_e, v_e, r_p, v_p, r_a, v_a = u[1:6]
	r_ee, r_xe, r_ep, r_xp, r_xa = u[[10, 22, 38, 42, 46]]
	r_s = u[47]
    η_e, η_p, η_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as, Δ_e, Δ_p, Δ_a = p

	# set/adjust parameters
	#######################

	k_ep_d = 5
	k_p_d = 4
	k_a_d = 4
	k_e_d = 4
	η_s = 0.002
	τ_s = 1.0

	τ_e = 13.0
	τ_p = 25.0
	τ_a = 20.0

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
    du[2] = (v_e^2 + η_e + (k_ee*r_ee - k_ep*r_ep)*τ_e - (τ_e*π*r_e)^2) / τ_e

    # GPe-p
    du[3] = (Δ_p/(π*τ_p) + 2.0*r_p*v_p) / τ_p
    du[4] = (v_p^2 + η_p + (k_pe*r_xe - k_pp*r_xp - k_pa*r_xa - k_ps*r_s)*τ_p - (τ_p*π*r_p)^2) / τ_p

    # GPe-a
    du[5] = (Δ_a/(π*τ_a) + 2.0*r_a*v_a) / τ_a
    du[6] = (v_a^2 + η_a + (k_ae*r_xe - k_ap*r_xp - k_aa*r_xa - k_as*r_s)*τ_a - (τ_a*π*r_a)^2) / τ_a

	# dummy str
	du[47] = (η_s - r_s) / τ_s

    # axonal propagation
    ####################

    # STN projections
	du[7] = k_e_d * (r_e - u[7])
	du[8] = k_e_d * (u[7] - u[8])
	du[9] = k_e_d * (u[8] - u[9])
	du[10] = k_e_d * (u[9] - u[10])
	du[11] = k_e_d * (u[10] - u[11])
	du[12] = k_e_d * (u[11] - u[12])
	du[13] = k_e_d * (u[12] - u[13])
	du[14] = k_e_d * (u[13] - u[14])
	du[15] = k_e_d * (u[14] - u[15])
	du[16] = k_e_d * (u[15] - u[16])
	du[17] = k_e_d * (u[16] - u[17])
	du[18] = k_e_d * (u[17] - u[18])
	du[19] = k_e_d * (u[18] - u[19])
	du[20] = k_e_d * (u[19] - u[20])
	du[21] = k_e_d * (u[20] - u[21])
	du[22] = k_e_d * (u[21] - u[22])

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

end

# initial condition and parameters
N = 47
u0 = zeros(N,)
tspan = [0., 100.]

#rng = MersenneTwister(1234)
# Δ_e = 0.1
# Δ_p = 0.3
# Δ_a = 0.2
#
# η_e = -1.0
# η_p = -0.8
# η_a = -1.6
#
# k_ee = 6.0
# k_pe = 95.0
# k_ae = 38.0
# k_ep = 19.0
# k_pp = 8.0
# k_ap = 29.0
# k_pa = 67.0
# k_aa = 0.0
# k_ps = 45.0
# k_as = 188.0
#
# p = [η_e, η_p, η_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as, Δ_e, Δ_p, Δ_a]
@load "BasalGanglia/results/test_params_0_params.jdl" p

# firing rate targets
targets=[[19, 62, 35],  # healthy control
         [missing, 35, missing],  # ampa blockade in GPe
         [missing, 76, missing],  # ampa and gabaa blockade in GPe
         [missing, 135, missing],  # GABAA blockade in GPe
         [38, 124, missing]  # GABAA blockade in STN
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
function stn_gpe_run(p)

	# calculate new
	for i=1:length(targets)

		# apply condition
		indices, k_scales = conditions[i]
		p_tmp = deepcopy(p)
		for (idx, k) in zip(indices, k_scales)
			p_tmp[idx] = p_tmp[idx] * k
		end

		# run simulation
		sol = Array(solve(remake(stn_gpe_prob, p=p_tmp), DP5(), saveat=0.1, reltol=1e-4, abstol=1e-6)) .* 1e3

		# plot results
		display(plot(sol[target_vars, :]'))
		sleep(5.0)

	end
end

# simulate conditions and plot them
stn_gpe_run(p)
