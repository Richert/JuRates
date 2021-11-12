using Distributions, Random, Statistics, PyCall, Distributed, DifferentialEquations, Plots, BlackBoxOptim, Interpolations

@pyimport pickle

function myunpickle(filename)
    r = nothing
    @pywith pybuiltin("open")(filename,"rb") as f begin
        r = pickle.load(f)
    end
    return r
end

# constant definitions and initializations
##########################################

# global constants
c = 1/π^2
target_vars = [1, 3, 5, 11]
rng = MersenneTwister(2234)
path = "/home/rgast/PycharmProjects/BrainNetworks/BasalGanglia/config/stn_gpe_str_config.pkl"
config = myunpickle(path)

# gamma-kernel convolution rates
m1 = 9
m2 = 6
m3 = 3
idx0 = 16
idx1 = idx0+m1
idx2 = idx1+m1
idx3 = idx2+m1
idx4 = idx3+m2
idx5 = idx4+m3

k_d1 = zeros(m1,)
k_d1[1] = 0.75
k_d1[2] = 1.0
k_d1[3] = 1.0
k_d1[4] = 2.0
k_d1[5] = 2.0
k_d1[6] = 2.0
k_d1[7] = 5/3
k_d1[8] = 0.72463768
k_d1[9] = 0.72463768

k_d2 = zeros(m1,)
k_d2[1] = 0.75
k_d2[2] = 1.0
k_d2[3] = 1.0
k_d2[4] = 2.0
k_d2[5] = 2.0
k_d2[6] = 2.0
k_d2[7] = 5/3
k_d2[8] = 0.72463768
k_d2[9] = 0.72463768

k_d3 = zeros(m1,)
k_d3[1] = 0.75
k_d3[2] = 1.0
k_d3[3] = 1.0
k_d3[4] = 2.0
k_d3[5] = 2.0
k_d3[6] = 2.0
k_d3[7] = 5/3
k_d3[8] = 0.72463768
k_d3[9] = 0.72463768

k_d4 = zeros(m2,)
k_d4[1] = 2.0
k_d4[2] = 2.0
k_d4[3] = 2.0
k_d4[4] = 5/3
k_d4[5] = 0.72463768
k_d4[6] = 0.72463768

k_d5 = zeros(m3,)
k_d5[1] = 5/3
k_d5[2] = 0.72463768
k_d5[3] = 0.72463768

r_out = zeros(m1,)

# stn-gpe-str constants
τ_s = 10.0
τ_p = 15.0
τ_a = 25.0
τ_d1 = 8.0
τ_d2 = 6.0
τ_f = 20.0

Δ_s = 1.0
Δ_p = 8.0
Δ_a = 5.0
Δ_d1 = 0.1
Δ_d2 = 0.2
Δ_f = 2.0

# definition of the initial value problem
#########################################

# state vector
N = idx5-1
u0 = zeros(N,)

# integration range in ms
T = 7000.0
tspan = [0., T]
dt = 0.1
dts = 1.0
n_steps = Int32(T/dt)
time = LinRange(tspan[1], T, n_steps)

# extrinsic input
cutoff_inp = Int32(1000/dt)
cutoff = Int32(1000/dts)
in_start = cutoff_inp + Int32(2000/dt)
in_stop = cutoff_inp + Int32(4000/dt)
inp = zeros(n_steps,)
inp[in_start:in_stop] .= 1.0
I_ext_interp = LinearInterpolation(time, inp)

# lower bounds
p_lower = [
			0.0, # η_s
			0.0, # η_p
			-5.0, # η_a
			-10.0, # η_d1
			-10.0, # η_d2
			-10.0, # η_f
			1.0, # k_sp
			10.0, # k_ps
			1.0, # k_pp
			20.0, # k_pd2
			5.0, # k_as
			1.0, # k_ap
			10.0, # k_ad1
			10.0, # k_d1a
			1.0, # k_d1d2
			10.0, # k_d1f
			10.0, # k_d2a
			1.0, # k_d2d1
			10.0, # k_d2f
			1.0, # k_fp
			1.0, # k_ff
			0.0, # α_a
			0.0, # α_d1
			0.0, # α_d2
			50.0, # τ_sfa
			0.0, # g_f
			0.01, # k_d2_exc
			-0.3, # k_s_inh
			0.001, # k_s_exc
		   ]

# upper bounds
p_upper = [
			20.0, # η_s
			20.0, # η_p
			15.0, # η_a
			10.0, # η_d1
			10.0, # η_d2
			10.0, # η_f
			60.0, # k_sp
			100.0, # k_ps
			40.0, # k_pp
			200.0, # k_pd2
			20.0, # k_as
			60.0, # k_ap
			100.0, # k_ad1
			100.0, # k_d1a
			50.0, # k_d1d2
			200.0, # k_d1f
			200.0, # k_d2a
			20.0, # k_d2d1
			150.0, # k_d2f
			50.0, # k_fp
			50.0, # k_ff
			5.0, # α_a
			10.0, # α_d1
			10.0, # α_d2
			300.0, # τ_sfa
			1.9, # g_f
			0.9, # k_d2_exc
			-0.01, # k_s_inh
			0.2, # k_s_exc
		   ]

function init_value(idx, var)
	return 0.5*(p_upper[idx] + p_lower[idx]) + var*randn(rng, Float32)*(p_upper[idx] - p_lower[idx])
end

# constants: excitabilities
init_var = 0.2
η_s = init_value(1, init_var)
η_p = init_value(2, init_var)
η_a = init_value(3, init_var)
η_d1 = init_value(4, init_var)
η_d2 = init_value(5, init_var)
η_f = init_value(6, init_var)

# constants: STN input scalings
k_si = 0.0
k_sp = init_value(7, init_var)

# constants: GPe-p input scalings
k_pi = 0.0
k_ps = init_value(8, init_var)
k_pp = init_value(9, init_var)
k_pd2 = init_value(10, init_var)

# constants: GPe-a input scalings
k_ai = 0.0
k_as = init_value(11, init_var)
k_ap = init_value(12, init_var)
k_ad1 = init_value(13, init_var)

# constants: MSN-D1 input scalings
k_d1i = 0.0
k_d1a = init_value(14, init_var)
k_d1d2 = init_value(15, init_var)
k_d1f = init_value(16, init_var)

# constants: MSN-D2 input scalings
k_d2i = 0.0
k_d2a = init_value(17, init_var)
k_d2d1 = init_value(18, init_var)
k_d2f = init_value(19, init_var)

# constants: FSI input scalings
k_fi = 0.0
k_fp = init_value(20, init_var)
k_ff = init_value(21, init_var)

# constants: SFA
α_a = init_value(22, init_var)
α_d1 = init_value(23, init_var)
α_d2 = init_value(24, init_var)
τ_sfa = init_value(25, init_var)

# constants: Gap junctions
g_f = init_value(26, init_var)

# constants: input strengths
k_d2_exc = init_value(27, init_var)
k_s_inh = init_value(28, init_var)
k_s_exc = init_value(29, init_var)

# parameter vectors
p0 = [η_s, η_p, η_a, η_d1, η_d2, η_f, k_sp, k_ps, k_pp, k_pd2, k_as, k_ap, k_ad1, k_d1a, k_d1d2, k_d1f, k_d2a, k_d2d1, k_d2f, k_fp, k_ff, α_a, α_d1, α_d2, τ_sfa, g_f, k_si, k_pi, k_ai, k_d1i, k_d2i, k_fi]
p1 = [η_s, η_p, η_a, η_d1, η_d2, η_f, k_sp, k_ps, k_pp, k_pd2, k_as, k_ap, k_ad1, k_d1a, k_d1d2, k_d1f, k_d2a, k_d2d1, k_d2f, k_fp, k_ff, α_a, α_d1, α_d2, τ_sfa, g_f, k_d2_exc, k_s_inh, k_s_exc]

# model definition
model = ODEProblem(stn_gpe_str, u0, tspan, p0)

# definiton of the different input conditions and their targets
targets = config["targets"]
conditions = [(31, 27), (27, 28), (27, 29)]
n_pars = length(p1)-3

# function definitions
######################

# vector field evaluation function
function stn_gpe_str(du, u, p, t)

    # preparations
	##############

	# extract state variables
    r_s, v_s, r_p, v_p, r_a, v_a, x_a, r_d1, v_d1, x_d1, r_d2, v_d2, x_d2, r_f, v_f = u[1:15]
	r_os, r_ip, r_oa = u[[idx2, idx2+1, idx2+2]]
	r_id1, r_id2, r_if = u[[idx3, idx3+1, idx3+2]]
	r_op, r_od1, r_od2 = u[[idx4, idx4+1, idx4+2]]

	# extract parameters
	η_s, η_p, η_a, η_d1, η_d2, η_f, k_sp, k_ps, k_pp, k_pd2, k_as, k_ap, k_ad1, k_d1a, k_d1d2, k_d1f, k_d2a, k_d2d1, k_d2f, k_fp, k_ff, α_a, α_d1, α_d2, τ_sfa, g_f, k_si, k_pi, k_ai, k_d1i, k_d2i, k_fi = p[:]

	# interpolate extrinsic input
	I_ext = I_ext_interp(t)

	# define synaptic inputs
	s_s = k_si*I_ext - k_sp*r_op
 	s_p = k_pi*I_ext + k_ps*r_os - k_pp*r_op - k_pd2*r_od2
	s_a = k_ai*I_ext + k_as*r_os - k_ap*r_ip - k_ad1*r_od1
	s_d1 = k_d1i*I_ext - k_d1a*r_oa - k_d1d2*r_id2 - k_d1f*r_if
	s_d2 = k_d2i*I_ext  - k_d2a*r_oa - k_d2d1*r_id1 - k_d2f*r_if
	s_f = k_fi*I_ext - k_fp*r_op - k_ff*r_if

    # populations
    #############

    # STN
    du[1] = (Δ_s/(π*τ_s) + 2.0*r_s*v_s) * (c/τ_s)
    du[2] = (v_s^2 + (η_s*c + s_s*τ_s - (τ_s*π*r_s)^2)/c^2) * (c/τ_s)

    # GPe-p
    du[3] = (Δ_p/(π*τ_p) + 2.0*r_p*v_p) * (c/τ_p)
    du[4] = (v_p^2 + (η_p*c + s_p*τ_p - (τ_p*π*r_p)^2)/c^2) * (c/τ_p)

    # GPe-a
    du[5] = (Δ_a/(π*τ_a) + 2.0*r_a*v_a) * (c/τ_a)
    du[6] = (v_a^2 + ((η_a - x_a)*c + s_a*τ_a - (τ_a*π*r_a)^2)/c^2) * (c/τ_a)
	du[7] = α_a*r_a - x_a/τ_sfa

	# MSN-D1
	du[8] = (Δ_d1/(π*τ_d1) + 2.0*r_d1*v_d1) * (c/τ_d1)
    du[9] = (v_d1^2 + ((η_d1 - x_d1)*c + s_d1*τ_d1 - (τ_d1*π*r_d1)^2)/c^2) * (c/τ_d1)
	du[10] = α_d1*r_d1 - x_d1/τ_sfa

	# MSN-D2
	du[11] = (Δ_d2/(π*τ_d2) + 2.0*r_d2*v_d2) * (c/τ_d2)
    du[12] = (v_d2^2 + ((η_d2 - x_d2)*c + s_d2*τ_d2 - (τ_d2*π*r_d2)^2)/c^2) * (c/τ_d2)
	du[13] = α_d2*r_d2 - x_d2/τ_sfa

	# FSI
	du[14] = (Δ_f/(π*τ_f) + r_f*(2*v_f-g_f)) * (c/τ_f)
    du[15] = (v_f^2 + (η_f*c + s_f*τ_f - (τ_f*π*r_f)^2)/c^2) * (c/τ_f)

	# synapse dynamics
    ##################

	# synaptic inputs
	r_out[1] = r_s
	r_out[2] = r_p
	r_out[3] = r_a
	r_out[4] = r_d1
	r_out[5] = r_d2
	r_out[6] = r_f
	r_out[7] = r_p
	r_out[8] = r_d1
	r_out[9] = r_d2

    # synaptic dynamics
	du[idx0:idx1-1] .= k_d1 .* (r_out .- u[idx0:idx1-1])
	du[idx1:idx2-1] .= k_d2 .* (u[idx0:idx1-1] .- u[idx1:idx2-1])
	du[idx2:idx3-1] .= k_d3 .* (u[idx1:idx2-1] .- u[idx2:idx3-1])
	du[idx3:idx4-1] .= k_d4 .* (u[idx2:idx3-1][m1-m2+1:end] .- u[idx3:idx4-1])
	du[idx4:idx5-1] .= k_d5 .* (u[idx3:idx4-1][m2-m3+1:end] .- u[idx4:idx5-1])

end

# summed squared error function
function sse(y, target)
    diff = y .- target
	#t_abs = abs.(target)
	#idx = t_abs .< 1.0
    #t_abs[idx] .= 1.0
    return diff' * diff #sum(diff./t_abs)
end

# IVP solver function iterating over different conditions
function run_all_conditions(conds, m, p)

    rates = []
    for (idx1, idx2) in conds

        # finalize new parameter vector
		p_tmp = p0[:]
		p_tmp[1:n_pars] = p[1:n_pars]
        p_tmp[idx1] = p[idx2]

        # perform simulation
		sol = Array(solve(remake(m, p=p_tmp), Tsit5(), saveat=dts, reltol=1e-8, abstol=1e-8))
        push!(rates, sol[target_vars, cutoff+1:end-1]' .* 1e3)

	end

    return rates
end

# loss function
function loss(p)

    # simulate system dynamics for different conditions
    solutions = run_all_conditions(conditions, model, p)

    # calculate loss
    loss = 0
    for (i, sol) in enumerate(solutions)
        for j = 1:size(targets)[1]
            t = targets[j, i]
            if t != nothing
				y = sol[:, j]
                loss += rmse(y, t)
			else
				continue
			end
		end
	end

    return loss
end

# callback function
fitness_progress_history = Array{Tuple{Int, Float64},1}()
cb = function(oc) #callback function to observe training
	p = best_candidate(oc)
    sols = run_all_conditions(conditions, model, p)
	p1 = plot(sols[1], label=["STN" "GPe-p" "GPe-a" "MSN-D2"], title="MSN-D2 excitation")
	p2 = plot(sols[2], label=["STN" "GPe-p" "GPe-a" "MSN-D2"], title="STN inhibition")
	p3 = plot(sols[3], label=["STN" "GPe-p" "GPe-a" "MSN-D2"], title="STN excitation")
	display(plot(p1, p2, p3, layout=(3,1)))
 return push!(fitness_progress_history, (BlackBoxOptim.num_func_evals(oc), best_fitness(oc)))
end

# Evolutionary optimization
###########################

# choose optimization algorithm
method = :dxnes

# start optimization
opt = bbsetup(loss; Method=method, Parameters=p1, SearchRange=(collect(zip(p_lower,p_upper))), NumDimensions=length(p1), MaxSteps=500, workers=workers(), TargetFitness=0.0, PopulationSize=1000, CallbackFunction=cb, CallbackInterval=1.0)
el = @elapsed res = bboptimize(opt)
t = round(el, digits=3)

# receive optimization results
p = best_candidate(res)
f = best_fitness(res)
display(p)

η_s, η_p, η_a, η_d1, η_d2, η_f, k_sp, k_ps, k_pp, k_pd2, k_as, k_ap, k_ad1, k_d1a, k_d1d2, k_d1f, k_d2a, k_d2d1, k_d2f, k_fp, k_ff, α_a, α_d1, α_d2, τ_sfa, g_f, k_d2_exc, k_s_inh, k_s_exc = p

# plot final results
sols = run_all_conditions(conditions, model, p)
p1 = plot(sols[1], label=["STN" "GPe-p" "GPe-a" "MSN-D2"], title="MSN-D2 excitation")
p2 = plot(sols[2], label=["STN" "GPe-p" "GPe-a" "MSN-D2"], title="STN inhibition")
p3 = plot(sols[3], label=["STN" "GPe-p" "GPe-a" "MSN-D2"], title="STN excitation")
p4 = plot(inp[cutoff_inp:end], title="Extrinsic input")
#p5 = plot(targets[:, 1], label=["STN" "GPe-p" "GPe-a" "MSN-D2"], title="Targets")
display(plot(p1, p2, p3, p4, layout=(4,1), size=(600, 600)))
