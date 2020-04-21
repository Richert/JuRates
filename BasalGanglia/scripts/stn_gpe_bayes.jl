using Plots, DifferentialEquations, DiffEqBayes

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
tmax = 100.0
tspan = [0., tmax]
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
k_pe = 188.4*√Δ_p
k_ae = 58.2*√Δ_a
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

# model definition and initial simulation
model = ODEProblem(stn_gpe, u0, tspan, p)
sol = solve(model, Tsit5())

# data collection
t = collect(range(10,stop=tmax,length=10))
randomized = VectorOfArray([(sol(t[i]) + .1randn(2)) for i in 1:length(t)])
data = convert(Array,randomized)

# prior definition
priors = [Normal(-2.0*Δ_e,4.0*Δ_e), # η_e
          Normal(-4.0*Δ_p,4.0*Δ_p), # η_p
          Normal(-8.0*Δ_a,4.0*Δ_a), # η_a
          Truncated(Normal(0.2*τ_e^2,0.2*τ_e^2),0.0,0.5*τ_e^2), # Δ_e
          Truncated(Normal(0.6*τ_p^2,0.3*τ_p^2),0.0,1.0*τ_p^2), # Δ_p
          Truncated(Normal(0.6*τ_a^2,0.3*τ_a^2),0.0,1.0*τ_a^2), # Δ_a
          Normal(13.0,0.1), # τ_e
          Normal(25,0.1), # τ_p
          Normal(20,0.1), # τ_a
          Uniform(0,50*√Δ_e), # k_ee
          Uniform(0,400*√Δ_p), # k_pe
          Uniform(0,400*√Δ_a), # k_ae
          Uniform(0,400*√Δ_e), # k_ep
          Uniform(0,200*√Δ_p), # k_pp
          Uniform(0,200*√Δ_a), # k_ap
          Uniform(0,200*√Δ_p), # k_pa
          Uniform(0,200*√Δ_a), # k_aa
          Uniform(0,600*√Δ_p), # k_ps
          Uniform(0,600*√Δ_a), # k_as
          ]

# perform bayesian inference
bayes = stan_inference(model,t,data,priors;
                       num_samples=100,num_warmup=500,
                       vars = (StanODEData(),InverseGamma(19,1)))

# show results
Mamba.describe(bayes.chain_results)
plot_chain(bayes)
