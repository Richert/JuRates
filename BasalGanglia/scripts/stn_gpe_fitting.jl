using Flux, DiffEqFlux, OrdinaryDiffEq, Plots, DiffEqSensitivity, Optim

# definition of the motion equations
function stn_gpe(du, u, p, t)

    # extract state vars and params
    r_e, v_e, r_p, v_p, r_a, v_a = u
    η_e, η_p, η_a, Δ_e, Δ_p, Δ_a, τ_e, τ_p, τ_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as = p

    # STN
    du[1] = (Δ_e/(π*τ_e) + 2.0*r_e*v_e) / τ_e
    du[2] = (v_e^2 + η_e + (k_ee*r_e - k_ep*r_p)*τ_e - (τ_e*π*r_e)^2) / τ_e

    # GPe-p
    du[3] = (Δ_p/(π*τ_p) + 2.0*r_p*v_p) / τ_p
    du[4] = (v_p^2 + η_p + (k_pe*r_e - k_pp*r_p - k_pa*r_a - k_ps*0.002)*τ_p - (τ_p*π*r_p)^2) / τ_e

    # GPe-a
    du[5] = (Δ_a/(π*τ_a) + 2.0*r_a*v_a) / τ_a
    du[6] = (v_a^2 + η_a + (k_ae*r_e - k_ap*r_p - k_aa*r_a - k_as*0.002)*τ_a - (τ_a*π*r_a)^2) / τ_a

end

# initial condition and parameters
u0 = zeros(6,)
tspan = [0., 100.]
#u0_f(p,t0) = [p[2],p[4]]
#tspan_f(p) = (0.0,10*p[4])

τ_e = 13.0
τ_p = 25.0
τ_a = 20.0

Δ_e = 0.347*τ_e^2
Δ_p = 0.593*τ_p^2
Δ_a = 0.778*τ_a^2

η_e = -1.48*Δ_e
η_p = -3.24*Δ_p
η_a = -13.20*Δ_a

k_ee = 25.5*√Δ_e
k_pe = 58.2*√Δ_p
k_ae = 188.4*√Δ_a
k_ep = 184.1*√Δ_e
k_pp = 42.6*√Δ_p
k_ap = 137.8*√Δ_a
k_pa = 58.8*√Δ_p
k_aa = 6.9*√Δ_a
k_ps = 585.8*√Δ_p
k_as = 126.0*√Δ_a

p = [η_e, η_p, η_a, Δ_e, Δ_p, Δ_a, τ_e, τ_p, τ_a, k_ee, k_pe, k_ae, k_ep, k_pp, k_ap, k_pa, k_aa, k_ps, k_as]

conditions = [([], []),  # healthy control
              ([11,12], [0.2,0.2]), # AMPA blockade in GPe
              ([11,12,14,15,16,17,18,19], [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]), # AMPA blockade and GABAA blockade in GPe
              ([14,15,16,17,18,19], [0.2,0.2,0.2,0.2,0.2,0.2]), # GABAA blockade in GPe
              ([11,12], [0.0,0.0]), # STN blockade
              ([11,12,14,15,16,17,18,19], [0.0,0.0,0.2,0.2,0.2,0.2,0.2,0.2]), # STN blockade + GABAA blockade in GPe
              ([13], [0.2]) # GABAA blocker in STN
             ]

# firing rate targets
targets=[[20, 60, 30]  # healthy control
         [NaN, 30, NaN]  # ampa blockade in GPe
         [NaN, 70, NaN]  # ampa and gabaa blockade in GPe
         [NaN, 100, NaN]  # GABAA blockade in GPe
         [NaN, 20, NaN]  # STN blockade
         [NaN, 60, NaN]  # STN blockade + gabaa blockade in GPe
         [40, 100, NaN]  # GABAA blockade in STN
        ]
#targets = [20, 60, 30]

n = length(conditions)

# model definition
model = ODEProblem(stn_gpe, u0, tspan, p)

# model simulation over conditions and calculation of loss
function stn_gpe_loss(p)

    # run simulations and calculate loss
    diff = []
    for i=1:n
        indices, scalings = conditions[i]
        p_tmp = deepcopy(p)
        for (idx, scale) in zip(indices, scalings)
            p_tmp[idx] *= scale
        end
        sol = concrete_solve(model,Tsit5(),u0,p_tmp,saveat=0.1,abstol=1e-6,
                             reltol=1e-6,sensealg=InterpolatingAdjoint()
                             )[[1,3,5], :]
        diff += sum([(s - t)^2 for (s, t) in zip(sol, targets[i]) if ! isnan(t)])
    end
    return diff
end

# model optimization
data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function (p,l) #callback function to observe training
  display(l)
  # using `remake` to re-create our `prob` with current parameters `p`
  display(plot(solve(remake(model,p=p),Tsit5(),saveat=0.1)[[1,3,5], :]'))
  #return false # Tell it to not halt the optimization. If return true, then optimization stops
end

# Display the ODE with the initial parameter values.
cb(p,stn_gpe_loss(p))

res = DiffEqFlux.sciml_train(stn_gpe_loss, p, BFGS(), cb = cb)
