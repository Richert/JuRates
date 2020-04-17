using DifferentialEquations,Plots,LSODA

# definition of the motion equations
function montbrio(du, u, p, t)
    r, v = u
    η, k, Δ, τ = p
    du[1] = (Δ/(π*τ) + 2.0*r*v) / τ
    du[2] = (v^2 + η + k*r*τ - τ*π*r) / τ
end

# initial condition and parameters
u0 = zeros(2)
u0[1] = 0.06
u0[2] = -2.0
tspan = [0., 100.]
τ = 13.0
Δ = 0.1*τ^2
η = -0.2*Δ
k = 1*√Δ
p = [η, k, Δ, τ]

# model setup and numerical solution
model = ODEProblem(montbrio, u0, tspan, p)
solution = solve(model, lsoda(), saveat=0.1)

# plotting
plot(solution[1, :])
