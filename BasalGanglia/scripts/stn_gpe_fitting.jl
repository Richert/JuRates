using Flux, DiffEqFlux, DifferentialEquations, Plots, DiffEqSensitivity, Optim, Statistics, JLD2, FileIO, Random, Distributions


# definition of the motion equations
τ_e = 13.0
τ_p = 25.0
τ_a = 20.0

function stn_gpe(du, u, p, t)

    # extract state vars and params
	###############################
    r_e1, v_e1, r_p1, v_p1, r_a1, v_a1, r_e2, v_e2, r_p2, v_p2, r_a2, v_a2,    r_e3, v_e3, r_p3, v_p3, r_a3, v_a3, r_e4, v_e4, r_p4, v_p4, r_a4, v_a4,    r_e5, v_e5, r_p5, v_p5, r_a5, v_a5 = u[1:30]
	r_pe1, r_pe2, r_pe3, r_pe4, r_pe5 = u[106:110]
	r_ep1, r_ep2, r_ep3, r_ep4, r_ep5 = u[186:190]
	r_xp1, r_xp2, r_xp3, r_xp4, r_xp5 = u[266:270]
	r_xa1, r_xa2, r_xa3, r_xa4, r_xa5 = u[346:350]
	r_ee1, r_ee2, r_ee3, r_ee4, r_ee5 = u[426:430]
    η_e, η_p, η_a, Δ_e, Δ_p, Δ_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as = p

	# set/adjust parameters
	#######################

    Δ_e = Δ_e*τ_e^2
    Δ_p = Δ_p*τ_p^2
    Δ_a = Δ_a*τ_a^2

    η_e = η_e*Δ_e*10
    η_p = η_p*Δ_p*10
    η_a = η_a*Δ_a*10

    k_ee = 100*k_ee*√Δ_e
    k_pe = 100*k_pe*√Δ_p
    k_ae = 100*k_ae*√Δ_a
    k_pp = 100*k_pp*√Δ_p
    k_ep = 100*k_ep*√Δ_e
    k_ap = 100*k_ap*√Δ_a
    k_pa = 100*k_pa*√Δ_p
    k_aa = 100*k_aa*√Δ_a
    k_ps = 100*k_ps*√Δ_p
    k_as = 100*k_as*√Δ_a

	r_e = [r_e1, r_e2, r_e3, r_e4, r_e5]
	r_p = [r_p1, r_p2, r_p3, r_p4, r_p5]
	r_a = [r_a1, r_a2, r_a3, r_a4, r_a5]

	k_pe_d = 4
	k_ep_d = 5
	k_p_d = 10
	k_a_d = 10
	k_ee_d = 10

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
    du[10] = (v_p2^2 + η_p + (0.2*k_pe*r_pe2 - k_pp*r_xp2 - k_pa*r_xa2 - k_ps*0.002)*τ_p - (τ_p*π*r_p2)^2) / τ_e

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
    du[16] = (v_p3^2 + η_p + 0.2*(k_pe*r_pe3 - k_pp*r_xp3 - k_pa*r_xa3 - k_ps*0.002)*τ_p - (τ_p*π*r_p3)^2) / τ_e

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
    du[22] = (v_p4^2 + η_p + 0.2*(k_pe*r_pe4*5.0 - k_pp*r_xp4 - k_pa*r_xa4 - k_ps*0.002)*τ_p - (τ_p*π*r_p4)^2) / τ_e

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
    du[28] = (v_p5^2 + η_p + (k_pe*r_pe5 - k_pp*r_xp5 - k_pa*r_xa5 - k_ps*0.002)*τ_p - (τ_p*π*r_p5)^2) / τ_e

    # GPe-a
    du[29] = (Δ_a/(π*τ_a) + 2.0*r_a5*v_a5) / τ_a
    du[30] = (v_a5^2 + η_a + (k_ae*r_pe5 - k_ap*r_xp5 - k_aa*r_xa5 - k_as*0.002)*τ_a - (τ_a*π*r_a5)^2) / τ_a

    # axonal propagation
    ####################

    # STN to GPe-p
	du[31:35] .= k_pe_d .* (r_e .- u[31:35])
	du[36:40] .= k_pe_d .* (u[31:35] .- u[36:40])
	du[41:45] .= k_pe_d .* (u[36:40] .- u[41:45])
	du[46:50] .= k_pe_d .* (u[41:45] .- u[46:50])
	du[51:55] .= k_pe_d .* (u[46:50] .- u[51:55])
	du[56:60] .= k_pe_d .* (u[51:55] .- u[56:60])
	du[61:65] .= k_pe_d .* (u[56:60] .- u[61:65])
	du[66:70] .= k_pe_d .* (u[61:65] .- u[66:70])
	du[71:75] .= k_pe_d .* (u[66:70] .- u[71:75])
	du[76:80] .= k_pe_d .* (u[71:75] .- u[76:80])
	du[81:85] .= k_pe_d .* (u[76:80] .- u[81:85])
	du[86:90] .= k_pe_d .* (u[81:85] .- u[86:90])
	du[91:95] .= k_pe_d .* (u[86:90] .- u[91:95])
	du[96:100] .= k_pe_d .* (u[91:95] .- u[96:100])
	du[101:105] .= k_pe_d .* (u[96:100] .- u[101:105])
	du[106:110] .= k_pe_d .* (u[101:105] .- u[106:110])

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
	du[211:215] .= k_p_d .* (u[206:210] .- u[211:215])
	du[216:220] .= k_p_d .* (u[211:215] .- u[216:220])
	du[221:225] .= k_p_d .* (u[216:220] .- u[221:225])
	du[226:230] .= k_p_d .* (u[221:225] .- u[226:230])
	du[231:235] .= k_p_d .* (u[226:230] .- u[231:235])
	du[236:240] .= k_p_d .* (u[231:235] .- u[236:240])
	du[241:245] .= k_p_d .* (u[236:240] .- u[241:245])
	du[246:250] .= k_p_d .* (u[241:245] .- u[246:250])
	du[251:255] .= k_p_d .* (u[246:250] .- u[251:255])
	du[256:260] .= k_p_d .* (u[251:255] .- u[256:260])
	du[261:265] .= k_p_d .* (u[256:260] .- u[261:265])
	du[266:270] .= k_p_d .* (u[261:265] .- u[266:270])

	# ! Gpe-a to both GPes
	du[271:275] .= k_a_d * (r_p .- u[271:275])
	du[276:280] .= k_a_d .* (u[271:275] .- u[276:280])
	du[281:285] .= k_a_d .* (u[276:280] .- u[281:285])
	du[286:290] .= k_a_d .* (u[281:285] .- u[286:290])
	du[291:295] .= k_a_d .* (u[286:290] .- u[291:295])
	du[296:300] .= k_a_d .* (u[291:295] .- u[296:300])
	du[301:305] .= k_a_d .* (u[296:300] .- u[301:305])
	du[306:310] .= k_a_d .* (u[301:305] .- u[306:310])
	du[311:315] .= k_a_d .* (u[306:310] .- u[311:315])
	du[316:320] .= k_a_d .* 190(u[311:315] .- u[316:320])
	du[321:325] .= k_a_d .* (u[316:320] .- u[321:325])
	du[326:330] .= k_a_d .* (u[321:325] .- u[326:330])
	du[331:335] .= k_a_d .* (u[326:330] .- u[331:335])
	du[336:340] .= k_a_d .* (u[331:335] .- u[336:340])
	du[341:345] .= k_a_d .* (u[336:340] .- u[341:345])
	du[346:350] .= k_a_d .* (u[341:345] .- u[346:350])

    # ! STN to STN
	du[351:355] .= k_ee_d * (r_p .- u[351:355])
	du[356:360] .= k_ee_d .* (u[351:355] .- u[356:360])
	du[361:365] .= k_ee_d .* (u[356:360] .- u[361:365])
	du[366:370] .= k_ee_d .* (u[361:365] .- u[366:370])
	du[371:375] .= k_ee_d .* (u[366:370] .- u[371:375])
	du[376:380] .= k_ee_d .* (u[371:375] .- u[376:380])
	du[381:385] .= k_ee_d .* (u[376:380] .- u[381:385])
	du[386:390] .= k_ee_d .* (u[381:385] .- u[386:390])
	du[391:395] .= k_ee_d .* (u[386:390] .- u[391:395])
	du[396:400] .= k_ee_d .* (u[391:395] .- u[396:400])
	du[401:405] .= k_ee_d .* (u[396:400] .- u[401:405])
	du[406:410] .= k_ee_d .* (u[401:405] .- u[406:410])
	du[411:415] .= k_ee_d .* (u[406:410] .- u[411:415])
	du[416:420] .= k_ee_d .* (u[411:415] .- u[416:420])
	du[421:425] .= k_ee_d .* (u[416:420] .- u[421:425])
	du[426:430] .= k_ee_d .* (u[421:425] .- u[426:430])

end

# initial condition and parameters
N = 430
N1 = 30
u0 = zeros(N,)
tspan = [0., 50.]

rng = MersenneTwister(1234)
Δ_e = rand(rng, truncated(Normal(0.25,0.1),0,0.5))
Δ_p = rand(rng, truncated(Normal(0.6,0.2),0,1.1))
Δ_a = rand(rng, truncated(Normal(0.6,0.2),0,1.1))

η_e = rand(rng, truncated(Normal(-0.1,0.1),-1.0,1.0))
η_p = rand(rng, truncated(Normal(-0.4,0.2),-1.0,1.0))
η_a = rand(rng, truncated(Normal(-0.8,0.2),-2.0,0.0))

k_ee = rand(rng, truncated(Normal(0.1,0.1),0,0.6))
k_pe = rand(rng, truncated(Normal(1.0,0.4),0,6.0))
k_ae = rand(rng, truncated(Normal(0.5,0.2),0,3.0))
k_ep = rand(rng, truncated(Normal(1.0,0.4),0,6.0))
k_pp = rand(rng, truncated(Normal(0.5,0.2),0,3.0))
k_ap = rand(rng, truncated(Normal(0.5,0.2),0,4.0))
k_pa = rand(rng, truncated(Normal(0.5,0.2),0,4.0))
k_aa = rand(rng, truncated(Normal(0.5,0.2),0,4.0))
k_ps = rand(rng, truncated(Normal(2.0,0.5),0,8.0))
k_as = rand(rng, truncated(Normal(3.0,0.5),0,8.0))

p = [η_e, η_p, η_a, Δ_e, Δ_p, Δ_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as]

#@load "BasalGanglia/results/stn_gpe_params.jld" p

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

    # run simulations and calculate loss
    if minimum(p[4:end]) < 0.0
        return 1e8
    else
        sol = Array(concrete_solve(stn_gpe_prob,BS5(),u0,p,saveat=0.1,sensealg=InterpolatingAdjoint(),reltol=1e-4,abstol=1e-6,maxiters=tspan[2]/1e-6,dt=1e-5,adaptive=true)) .* 1e3
        if size(sol)[2] < (tspan[2]-1)/0.1
            return 1e8
		elseif maximum(maximum(abs.(sol[target_vars, :]))) > 1000.0
			return maximum(maximum(abs.(sol[target_vars, :])))^2
        else
            diff1 = sum(abs2, s-t for (s,t) in zip(mean(sol[target_vars, :],dims=2), targets) if ! ismissing(t))
            diff2 = sum(ismissing(t) ? 1/var(sol[i, 20:200]) : var(sol[i, 50:end]) for (i, t) in zip(freq_indices, freq_targets))
            return diff1 + diff2
        end
    end
end

# model optimization
cb = function (p,l) #callback function to observe training
  display(l)
  # using `remake` to re-create our `prob` with current parameters `p`
  display(plot(solve(remake(stn_gpe_prob,p=p),BS5(),saveat=0.1,reltol=1e-4,abstol=1e-6,maxiters=tspan[2]/1e-6,dt=1e-5,adaptive=true)[target_vars, :]', ylims=[0.0, 0.2]))
  return false # Tell it to not halt the optimization. If return true, then optimization stops
end

# Display the ODE with the initial parameter values.
cb(p,stn_gpe_loss(p))

res = DiffEqFlux.sciml_train(stn_gpe_loss,p,ADAGrad(0.02),cb = cb, maxiters=3)
p_new = res.minimizer
display(p_new)
η_e, η_p, η_a, Δ_e, Δ_p, Δ_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as = p_new

jname = ARGS[1]
jid = ARGS[2]
@save "../results/$jname" * "_$jid" * "_params.jdl" p_new
