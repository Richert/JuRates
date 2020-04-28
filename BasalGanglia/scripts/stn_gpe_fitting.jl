using Distributions, Random, Statistics, FileIO, JLD2, Flux, Optim, DifferentialEquations, DiffEqSensitivity, DiffEqFlux, Plots

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
    r_e1, v_e1, r_p1, v_p1, r_a1, v_a1, r_e2, v_e2, r_p2, v_p2, r_a2, v_a2, r_e3, v_e3, r_p3, v_p3, r_a3, v_a3, r_e4, v_e4, r_p4, v_p4, r_a4, v_a4, r_e5, v_e5, r_p5, v_p5, r_a5, v_a5 = u[1:30]
	r_pe1, r_pe2, r_pe3, r_pe4, r_pe5 = u[106:110]
	r_ep1, r_ep2, r_ep3, r_ep4, r_ep5 = u[186:190]
	r_xp1, r_xp2, r_xp3, r_xp4, r_xp5 = u[206:210]
	r_xa1, r_xa2, r_xa3, r_xa4, r_xa5 = u[226:230]
	r_ee1, r_ee2, r_ee3, r_ee4, r_ee5 = u[46:50]
    η_e, η_p, η_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as = p

	# set/adjust parameters
	#######################

	r_e = [r_e1, r_e2, r_e3, r_e4, r_e5]
	r_p = [r_p1, r_p2, r_p3, r_p4, r_p5]
	r_a = [r_a1, r_a2, r_a3, r_a4, r_a5]

	k_e_d = 4
	k_ep_d = 5
	k_p_d = 4
	k_a_d = 4

	# η_e = η_e*Δ_e*10.0
	# η_p = η_p*Δ_p*10.0
	# η_a = η_a*Δ_a*10.0
	#
	# k_ee = k_ee*√Δ_e*100.0
	# k_pe = k_pe*√Δ_p*100.0
	# k_ae = k_ae*√Δ_a*100.0
	# k_pp = k_pp*√Δ_p*100.0
	# k_ep = k_ep*√Δ_e*100.0
	# k_ap = k_ap*√Δ_a*100.0
	# k_pa = k_pa*√Δ_p*100.0
	# k_aa = k_aa*√Δ_a*100.0
	# k_ps = k_ps*√Δ_p*100.0
	# k_as = k_as*√Δ_a*100.0

    # condition 1
    #############

    # STN
    du[1] = (Δ_e/(π*τ_e) + 2.0*r_e1*v_e1) / τ_e
    du[2] = (v_e1^2 + η_e + (k_ee*r_ee1 - k_ep*r_ep1)*τ_e - (τ_e*π*r_e1)^2) / τ_e

    # GPe-p
    du[3] = (Δ_p/(π*τ_p) + 2.0*r_p1*v_p1) / τ_p
    du[4] = (v_p1^2 + η_p + (k_pe*r_pe1 - k_pp*r_xp1 - k_pa*r_xa1 - k_ps*0.002)*τ_p - (τ_p*π*r_p1)^2) / τ_e

    # GPe-a
    du[5] = (Δ_a/(π*τ_a) + 2.0*r_a1*v_a1) / τ_a
    du[6] = (v_a1^2 + η_a + (k_ae*r_pe1 - k_ap*r_xp1 - k_aa*r_xa1 - k_as*0.002)*τ_a - (τ_a*π*r_a1)^2) / τ_a

    # condition 2
    #############

    # STN
    du[7] = (Δ_e/(π*τ_e) + 2.0*r_e2*v_e2) / τ_e
    du[8] = (v_e2^2 + η_e + (k_ee*r_ee2 - k_ep*r_ep2)*τ_e - (τ_e*π*r_e2)^2) / τ_e

    # GPe-p
    du[9] = (Δ_p/(π*τ_p) + 2.0*r_p2*v_p2) / τ_p
    du[10] = (v_p2^2 + η_p + (0.2*k_pe*r_pe2 - k_pp*r_xp2 - k_pa*r_xa2 - k_ps*0.002)*τ_p - (τ_p*π*r_p2)^2) / τ_p

    # GPe-a
    du[11] = (Δ_a/(π*τ_a) + 2.0*r_a2*v_a2) / τ_a
    du[12] = (v_a2^2 + η_a + (0.2*k_ae*r_pe2 - k_ap*r_xp2 - k_aa*r_xa2 - k_as*0.002)*τ_a - (τ_a*π*r_a2)^2) / τ_a

    # condition 3
    #############

    # STN
    du[13] = (Δ_e/(π*τ_e) + 2.0*r_e3*v_e3) / τ_e
    du[14] = (v_e3^2 + η_e + (k_ee*r_ee3 - k_ep*r_ep3)*τ_e - (τ_e*π*r_e3)^2) / τ_e

    # GPe-p
    du[15] = (Δ_p/(π*τ_p) + 2.0*r_p3*v_p3) / τ_p
    du[16] = (v_p3^2 + η_p + 0.2*(k_pe*r_pe3 - k_pp*r_xp3 - k_pa*r_xa3 - k_ps*0.002)*τ_p - (τ_p*π*r_p3)^2) / τ_p

    # GPe-a
    du[17] = (Δ_a/(π*τ_a) + 2.0*r_a3*v_a3) / τ_a
    du[18] = (v_a3^2 + η_a + 0.2*(k_ae*r_pe3 - k_ap*r_xp3 - k_aa*r_xa3 - k_as*0.002)*τ_a - (τ_a*π*r_a3)^2) / τ_a

    # condition 4
    #############

    # STN
    du[19] = (Δ_e/(π*τ_e) + 2.0*r_e4*v_e4) / τ_e
    du[20] = (v_e4^2 + η_e + (k_ee*r_ee4 - k_ep*r_ep4)*τ_e - (τ_e*π*r_e4)^2) / τ_e

    # GPe-p
    du[21] = (Δ_p/(π*τ_p) + 2.0*r_p4*v_p4) / τ_p
    du[22] = (v_p4^2 + η_p + 0.2*(k_pe*r_pe4*5.0 - k_pp*r_xp4 - k_pa*r_xa4 - k_ps*0.002)*τ_p - (τ_p*π*r_p4)^2) / τ_p

    # GPe-a
    du[23] = (Δ_a/(π*τ_a) + 2.0*r_a4*v_a4) / τ_a
    du[24] = (v_a4^2 + η_a + 0.2*(k_ae*r_pe4*5.0 - k_ap*r_xp4 - k_aa*r_xa4 - k_as*0.002)*τ_a - (τ_a*π*r_a4)^2) / τ_a

    # condition 5
    #############

    # STN
    du[25] = (Δ_e/(π*τ_e) + 2.0*r_e5*v_e5) / τ_e
    du[26] = (v_e5^2 + η_e + (k_ee*r_ee5 - 0.2*k_ep*r_ep5)*τ_e - (τ_e*π*r_e5)^2) / τ_e

    # GPe-p
    du[27] = (Δ_p/(π*τ_p) + 2.0*r_p5*v_p5) / τ_p
    du[28] = (v_p5^2 + η_p + (k_pe*r_pe5 - k_pp*r_xp5 - k_pa*r_xa5 - k_ps*0.002)*τ_p - (τ_p*π*r_p5)^2) / τ_p

    # GPe-a
    du[29] = (Δ_a/(π*τ_a) + 2.0*r_a5*v_a5) / τ_a
    du[30] = (v_a5^2 + η_a + (k_ae*r_pe5 - k_ap*r_xp5 - k_aa*r_xa5 - k_as*0.002)*τ_a - (τ_a*π*r_a5)^2) / τ_a

    # axonal propagation
    ####################

    # STN to GPe-p
	du[31:35] .= k_e_d .* (r_e .- u[31:35])
	du[36:40] .= k_e_d .* (u[31:35] .- u[36:40])
	du[41:45] .= k_e_d .* (u[36:40] .- u[41:45])
	du[46:50] .= k_e_d .* (u[41:45] .- u[46:50])
	du[51:55] .= k_e_d .* (u[46:50] .- u[51:55])
	du[56:60] .= k_e_d .* (u[51:55] .- u[56:60])
	du[61:65] .= k_e_d .* (u[56:60] .- u[61:65])
	du[66:70] .= k_e_d .* (u[61:65] .- u[66:70])
	du[71:75] .= k_e_d .* (u[66:70] .- u[71:75])
	du[76:80] .= k_e_d .* (u[71:75] .- u[76:80])
	du[81:85] .= k_e_d .* (u[76:80] .- u[81:85])
	du[86:90] .= k_e_d .* (u[81:85] .- u[86:90])
	du[91:95] .= k_e_d .* (u[86:90] .- u[91:95])
	du[96:100] .= k_e_d .* (u[91:95] .- u[96:100])
	du[101:105] .= k_e_d .* (u[96:100] .- u[101:105])
	du[106:110] .= k_e_d .* (u[101:105] .- u[106:110])

	# GPe-p to STN
	du[111:115] .= k_ep_d .* (r_p - u[111:115])
	du[116:120] .= k_ep_d .* (u[111:115] .- u[116:120])
	du[121:125] .= k_ep_d .* (u[116:120] .- u[121:125])
	du[126:130] .= k_ep_d .* (u[121:125] .- u[126:130])
	du[131:135] .= k_ep_d .* (u[126:130] .- u[131:135])
	du[136:140] .= k_ep_d .* (u[131:135] .- u[136:140])
	du[141:145] .= k_ep_d .* (u[136:140] .- u[141:145])
	du[146:150] .= k_ep_d .* (u[141:145] .- u[146:150])
	du[151:155] .= k_ep_d .* (u[146:150] .- u[151:155])
	du[156:160] .= k_ep_d .* (u[151:155] .- u[156:160])
	du[161:165] .= k_ep_d .* (u[156:160] .- u[161:165])
	du[166:170] .= k_ep_d .* (u[161:165] .- u[166:170])
	du[171:175] .= k_ep_d .* (u[166:170] .- u[171:175])
	du[176:180] .= k_ep_d .* (u[171:175] .- u[176:180])
	du[181:185] .= k_ep_d .* (u[176:180] .- u[181:185])
	du[186:190] .= k_ep_d .* (u[181:185] .- u[186:190])

	# Gpe-p to both GPes
	du[191:195] .= k_p_d * (r_p .- u[191:195])
	du[196:200] .= k_p_d .* (u[191:195] .- u[196:200])
	du[201:205] .= k_p_d .* (u[196:200] .- u[201:205])
	du[206:210] .= k_p_d .* (u[201:205] .- u[206:210])

	# ! Gpe-a to both GPes
	du[211:215] .= k_a_d * (r_a .- u[211:215])
	du[216:220] .= k_a_d .* (u[211:215] .- u[216:220])
	du[221:225] .= k_a_d .* (u[216:220] .- u[221:225])
	du[226:230] .= k_a_d .* (u[221:225] .- u[226:230])

end

# initial condition and parameters
N = 230
N1 = 30
u0 = zeros(N,)
tspan = [0., 50.]

#rng = MersenneTwister(1234)
#Δ_e = rand(truncated(Normal(0.05,0.01),0.02,0.08))
#Δ_p = rand(truncated(Normal(0.6,0.1),0.3,0.9))
#Δ_a = rand(truncated(Normal(0.3,0.1),0.1,0.5))

# η_e = rand(truncated(Normal(-0.1,0.05),-1.0,0.5))
# η_p = rand(truncated(Normal(-0.4,0.1),-1.0,0.5))
# η_a = rand(truncated(Normal(-0.8,0.2),-2.0,0.0))
#
# k_ee = rand(truncated(Normal(0.02,0.01),0,0.06))
# k_pe = rand(truncated(Normal(2.0,0.1),0.2,4.0))
# k_ae = rand(truncated(Normal(0.4,0.05),0.1,2.0))
# k_ep = rand(truncated(Normal(0.3,0.05),0.1,0.6))
# k_pp = rand(truncated(Normal(0.07,0.01),0.05,0.09))
# k_ap = rand(truncated(Normal(0.2,0.05),0.05,0.8))
# k_pa = rand(truncated(Normal(0.2,0.05),0.05,0.8))
# k_aa = rand(truncated(Normal(0.05,0.01),0.02,0.08))
# k_ps = rand(truncated(Normal(1.0,0.1), 0.4,4.0))
# k_as = rand(truncated(Normal(1.0,0.1), 0.4,4.0))

#Δ_e = Δ_e*τ_e^2
#Δ_p = Δ_p*τ_p^2
#Δ_a = Δ_a*τ_a^2

#p = [η_e, η_p, η_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as]

jname = ARGS[1]
jid = ARGS[2]
@load "results/$jname" * "_$jid" * "_params.jdl" p

# firing rate targets
targets=[[20, 60, 30]  # healthy control
         [missing, 30, missing]  # ampa blockade in GPe
         [missing, 70, missing]  # ampa and gabaa blockade in GPe
         [missing, 100, missing]  # GABAA blockade in GPe
         [40, 100, missing]  # GABAA blockade in STN
        ]
# oscillation behavior targets
freq_targets = [0.0, 0.0, missing, 0.0, missing]
freq_indices = 3:6:N1

# model definition
stn_gpe_prob = ODEProblem(stn_gpe, u0, tspan, p)
target_vars = 1:2:N1

# model simulation over conditions and calculation of loss
function stn_gpe_loss(p)

    # run simulation
    sol = Array(concrete_solve(stn_gpe_prob, DP5(), u0, p, sensealg=QuadratureAdjoint(), saveat=0.1, reltol=1e-4, abstol=1e-6, maxiters=tspan[2]/1e-6, dt=1e-5, adaptive=true)) .* 1e3

	# calculate loss
	diff1 = sum((s-t)^2/t for (s,t) in zip(mean(sol[target_vars, 400:end],dims=2), targets) if ! ismissing(t))
    diff2 = sum(ismissing(t) ? 1/var(sol[i, 50:end]) : var(sol[i, 50:end]) for (i, t) in zip(freq_indices, freq_targets))
	r_max = maximum(maximum(abs.(sol[target_vars, 200:end])))
	r_max^2 > 1000.0 ? diff3 = r_max^2 : diff3 = 0.0
	any(p[4:end] .< 0.0) ? diff4 = 1e6 : diff4 = 0.0
    return diff1 + diff2 + diff3 + diff4

end

# model optimization
cb = function (p,l) #callback function to observe training
  display(l)
  # using `remake` to re-create our `prob` with current parameters `p`
  if all(p[4:end] .> 0.0)
	  display(plot(solve(remake(stn_gpe_prob,p=p), DP5(), saveat=0.1, reltol=1e-4, abstol=1e-6, maxiters=tspan[2]/1e-6, dt=1e-5, adaptive=true)[target_vars, :]', ylims=[0.0, 0.2]))
  end
  return false # Tell it to not halt the optimization. If return true, then optimization stops
end

# Display the ODE with the initial parameter values.
cb(p,stn_gpe_loss(p))

# choose optimization algorithm
opt = ADAGrad(0.5)

# start optimization
res = DiffEqFlux.sciml_train(stn_gpe_loss, p, opt, cb=cb, maxiters=1000)

# receive optimization results
p = res.minimizer
display(p)
η_e, η_p, η_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as = p

# store best parameter set
@save "results/$jname" * "_$jid" * "_params_final.jdl" p
