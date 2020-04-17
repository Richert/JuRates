using DifferentialEquations, Plots, LSODA

# plotting backend
gr()

# definition of the motion equations
function montbrio(du, u, p, t)
    r, v = u
    η, k, Δ, τ = p
    du[1] = (Δ/(π*τ) + 2.0*r*v) / τ
    du[2] = (v^2 + η + k*r*τ - τ*π*r) / τ
end

# initial condition and parameters
u0 = zeros(2)
tspan = [0., 500.]

τ = 13.0
Δ = 0.1*τ^2
η = -2*Δ
k = 2*√Δ
p = [η, k, Δ, τ]

# definition of the parameter sweep
n = 100
η_all = range(-10.0*Δ, stop=0.0, length=n)
function sweep(prob,i,repeat)
  η_new = η_all[i]
  remake(prob,p=[[η_new,]; p[2:end]])
end

# model setup and numerical solution
model = ODEProblem(montbrio, u0, tspan, p)
model_sweep = EnsembleProblem(model,prob_func=sweep,
                              output_func=(solution, i) -> (solution[1, end], false))
solution = solve(model_sweep, lsoda(), saveat=0.1, trajectories=n)

# plotting
plot(η_all, solution[:])
