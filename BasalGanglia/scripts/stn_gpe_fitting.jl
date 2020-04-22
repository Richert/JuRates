using Flux, DiffEqFlux, DifferentialEquations, Plots, DiffEqSensitivity, Optim, Statistics


# definition of the motion equations
τ_e = 13.0
τ_p = 25.0
τ_a = 20.0

function stn_gpe(du, u, p, t)

    # extract state vars and params
    r_e1, v_e1, r_p1, v_p1, r_a1, v_a1, r_e2, v_e2, r_p2, v_p2, r_a2, v_a2,
    r_e3, v_e3, r_p3, v_p3, r_a3, v_a3, r_e4, v_e4, r_p4, v_p4, r_a4, v_a4,
    r_e5, v_e5, r_p5, v_p5, r_a5, v_a5, r_e6, v_e6, r_p6, v_p6, r_a6, v_a6,
    r_e7, v_e7, r_p7, v_p7, r_a7, v_a7 = u
    η_e, η_p, η_a, Δ_e, Δ_p, Δ_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as = p

    # condition 1
    #############

    # STN
    du[1] = (Δ_e/(π*τ_e) + 2.0*r_e1*v_e1) / τ_e
    du[2] = (v_e1^2 + η_e + (k_ee*r_e1 - k_ep*r_p1)*τ_e - (τ_e*π*r_e1)^2) / τ_e

    # GPe-p
    du[3] = (Δ_p/(π*τ_p) + 2.0*r_p1*v_p1) / τ_p
    du[4] = (v_p1^2 + η_p + (k_pe*r_e1 - k_pp*r_p1 - k_pa*r_a1 - k_ps*0.002)*τ_p - (τ_p*π*r_p1)^2) / τ_e

    # GPe-a
    du[5] = (Δ_a/(π*τ_a) + 2.0*r_a1*v_a1) / τ_a
    du[6] = (v_a1^2 + η_a + (k_ae*r_e1 - k_ap*r_p1 - k_aa*r_a1 - k_as*0.002)*τ_a - (τ_a*π*r_a1)^2) / τ_a

    # condition 2
    #############

    # STN
    du[7] = (Δ_e/(π*τ_e) + 2.0*r_e2*v_e2) / τ_e
    du[8] = (v_e2^2 + η_e + (k_ee*r_e2 - k_ep*r_p2)*τ_e - (τ_e*π*r_e2)^2) / τ_e

    # GPe-p
    du[9] = (Δ_p/(π*τ_p) + 2.0*r_p2*v_p2) / τ_p
    du[10] = (v_p2^2 + η_p + (0.2*k_pe*r_e2 - k_pp*r_p2 - k_pa*r_a2 - k_ps*0.002)*τ_p - (τ_p*π*r_p2)^2) / τ_e

    # GPe-a
    du[11] = (Δ_a/(π*τ_a) + 2.0*r_a2*v_a2) / τ_a
    du[12] = (v_a2^2 + η_a + (0.2*k_ae*r_e2 - k_ap*r_p2 - k_aa*r_a2 - k_as*0.002)*τ_a - (τ_a*π*r_a2)^2) / τ_a

    # condition 3
    #############

    # STN
    du[13] = (Δ_e/(π*τ_e) + 2.0*r_e3*v_e3) / τ_e
    du[14] = (v_e3^2 + η_e + (k_ee*r_e3 - k_ep*r_p3)*τ_e - (τ_e*π*r_e3)^2) / τ_e

    # GPe-p
    du[15] = (Δ_p/(π*τ_p) + 2.0*r_p3*v_p3) / τ_p
    du[16] = (v_p3^2 + η_p + 0.2*(k_pe*r_e3 - k_pp*r_p3 - k_pa*r_a3 - k_ps*0.002)*τ_p - (τ_p*π*r_p3)^2) / τ_e

    # GPe-a
    du[17] = (Δ_a/(π*τ_a) + 2.0*r_a3*v_a3) / τ_a
    du[18] = (v_a3^2 + η_a + 0.2*(k_ae*r_e3 - k_ap*r_p3 - k_aa*r_a3 - k_as*0.002)*τ_a - (τ_a*π*r_a3)^2) / τ_a

    # condition 4
    #############

    # STN
    du[19] = (Δ_e/(π*τ_e) + 2.0*r_e4*v_e4) / τ_e
    du[20] = (v_e4^2 + η_e + (k_ee*r_e4 - k_ep*r_p4)*τ_e - (τ_e*π*r_e4)^2) / τ_e

    # GPe-p
    du[21] = (Δ_p/(π*τ_p) + 2.0*r_p4*v_p4) / τ_p
    du[22] = (v_p4^2 + η_p + 0.2*(k_pe*r_e4*5.0 - k_pp*r_p4 - k_pa*r_a4 - k_ps*0.002)*τ_p - (τ_p*π*r_p4)^2) / τ_e

    # GPe-a
    du[23] = (Δ_a/(π*τ_a) + 2.0*r_a4*v_a4) / τ_a
    du[24] = (v_a4^2 + η_a + 0.2*(k_ae*r_e4*5.0 - k_ap*r_p4 - k_aa*r_a4 - k_as*0.002)*τ_a - (τ_a*π*r_a4)^2) / τ_a

    # condition 5
    #############

    # STN
    du[25] = (Δ_e/(π*τ_e) + 2.0*r_e5*v_e5) / τ_e
    du[26] = (v_e5^2 + η_e + (k_ee*r_e5 - k_ep*r_p5)*τ_e - (τ_e*π*r_e5)^2) / τ_e

    # GPe-p
    du[27] = (Δ_p/(π*τ_p) + 2.0*r_p5*v_p5) / τ_p
    du[28] = (v_p5^2 + η_p + (k_pp*r_p5 - k_pa*r_a5 - k_ps*0.002)*τ_p - (τ_p*π*r_p5)^2) / τ_e

    # GPe-a
    du[29] = (Δ_a/(π*τ_a) + 2.0*r_a5*v_a5) / τ_a
    du[30] = (v_a5^2 + η_a + (k_ap*r_p5 - k_aa*r_a5 - k_as*0.002)*τ_a - (τ_a*π*r_a5)^2) / τ_a

    # condition 6
    #############

    # STN
    du[31] = (Δ_e/(π*τ_e) + 2.0*r_e6*v_e6) / τ_e
    du[32] = (v_e6^2 + η_e + (k_ee*r_e6 - k_ep*r_p6)*τ_e - (τ_e*π*r_e6)^2) / τ_e

    # GPe-p
    du[33] = (Δ_p/(π*τ_p) + 2.0*r_p6*v_p6) / τ_p
    du[34] = (v_p6^2 + η_p + 0.2*(k_pp*r_p6 - k_pa*r_a6 - k_ps*0.002)*τ_p - (τ_p*π*r_p6)^2) / τ_e

    # GPe-a
    du[35] = (Δ_a/(π*τ_a) + 2.0*r_a6*v_a6) / τ_a
    du[36] = (v_a6^2 + η_a + 0.2*(k_ap*r_p6 - k_aa*r_a6 - k_as*0.002)*τ_a - (τ_a*π*r_a6)^2) / τ_a

    # condition 7
    #############

    # STN
    du[37] = (Δ_e/(π*τ_e) + 2.0*r_e7*v_e7) / τ_e
    du[38] = (v_e7^2 + η_e + (k_ee*r_e7 - 0.2*k_ep*r_p7)*τ_e - (τ_e*π*r_e7)^2) / τ_e

    # GPe-p
    du[39] = (Δ_p/(π*τ_p) + 2.0*r_p7*v_p7) / τ_p
    du[40] = (v_p7^2 + η_p + (k_pe*r_e7 - k_pp*r_p7 - k_pa*r_a7 - k_ps*0.002)*τ_p - (τ_p*π*r_p7)^2) / τ_e

    # GPe-a
    du[41] = (Δ_a/(π*τ_a) + 2.0*r_a7*v_a7) / τ_a
    du[42] = (v_a7^2 + η_a + (k_ae*r_e7 - k_ap*r_p7 - k_aa*r_a7 - k_as*0.002)*τ_a - (τ_a*π*r_a7)^2) / τ_a

end

# initial condition and parameters
u0 = zeros(42,)
tspan = [0., 50.]

Δ_e = 0.1*τ_e^2
Δ_p = 0.7*τ_p^2
Δ_a = 0.8*τ_a^2

η_e = -1*Δ_e
η_p = -4*Δ_p
η_a = -8*Δ_a

k_ee = 10*√Δ_e
k_pe = 100*√Δ_p
k_ae = 150*√Δ_a
k_ep = 200*√Δ_e
k_pp = 40*√Δ_p
k_ap = 100*√Δ_a
k_pa = 200*√Δ_p
k_aa = 200*√Δ_a
k_ps = 200*√Δ_p
k_as = 400*√Δ_a

# Δ_e = 25.3
# Δ_p = 301.2
# Δ_a = 271.0
#
# η_e = -112.2
# η_p = -653.5
# η_a = -1662.4
#
# k_ee = 243.8
# k_pe = 2365.0
# k_ae = 200.0
# k_ep = 275.7
# k_pp = 276.0
# k_ap = 2330.3
# k_pa = 1684.7
# k_aa = 532.3
# k_ps = 2702.6
# k_as = 2679.5

p = [η_e, η_p, η_a, Δ_e, Δ_p, Δ_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as]

# firing rate targets
targets=[[20, 60, 30]  # healthy control
         [missing, 30, missing]  # ampa blockade in GPe
         [missing, 70, missing]  # ampa and gabaa blockade in GPe
         [missing, 100, missing]  # GABAA blockade in GPe
         [40, 100, missing]  # GABAA blockade in STN
        ]
# oscillation behavior targets
freq_targets = [0.0, 0.0, missing, 0.0, missing]
freq_indices = 3:6:
# model definition
stn_gpe_prob = ODEProblem(stn_gpe, u0, tspan, p)
target_vars = 1:2:42

# model simulation over conditions and calculation of loss
function stn_gpe_loss(p)

    # run simulations and calculate loss
    sol = Array(concrete_solve(stn_gpe_prob,Tsit5(),u0,p,saveat=0.1,abstol=1e-6,
        reltol=1e-6, sensealg=BacksolveAdjoint())[target_vars, :]) .* 1e3
    diff1 = sum(abs2, s-t for (s,t) in zip(sol[:, end], targets) if ! ismissing(t))
    diff2 = sum(ismissing(t) ? 1/var(s))
    max_rate = maximum(maximum(sol))
    max_rate > 1000.0 ? diff3 = abs2(max_rate) : diff3 = 0.0
    minimum(p[4:end]) < 0.0 ? diff4 = 1e6 : diff4 = 0.0
    return diff1 + diff2 + diff3
end

# model optimization
cb = function (p,l) #callback function to observe training
  display(l)
  # using `remake` to re-create our `prob` with current parameters `p`
  display(plot(solve(remake(stn_gpe_prob,p=p),Tsit5(),saveat=0.1)[target_vars, :]', ylims=[0.0, 0.2]))
  return false # Tell it to not halt the optimization. If return true, then optimization stops
end

# Display the ODE with the initial parameter values.
cb(p,stn_gpe_loss(p))

res = DiffEqFlux.sciml_train(stn_gpe_loss, p, ADAGrad(0.1), cb = cb,
                             maxiters=5000)
p_new = res.minimizer
display(p_new)
η_e, η_p, η_a, Δ_e, Δ_p, Δ_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as = p_new
