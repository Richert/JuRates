using Flux, Plots

# definition of the motion equations
function rate_neuron(h, x)
    h .+= 0.001 .* (.-h .+ 21.3 * sum(1 ./(10.0 .* (0.1 .- x))))
    return h, h
end

# initial condition and parameters
N = 1000
global h = randn(N)
global x = rand(N)

# forward euler integration
solutions = []
steps = 10000
for i=1:steps
    results = rate_neuron(h, x)
    global h = results[1]
    push!(solutions, results[1])
end

# plotting
plot(solutions)
