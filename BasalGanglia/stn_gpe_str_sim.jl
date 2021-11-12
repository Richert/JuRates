using DifferentialEquations, Plots, Interpolations

# constant definitions and initializations
##########################################

# global constants
c = 1/π^2

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
τ_s = 12.0
τ_p = 15.0
τ_a = 23.0
τ_d1 = 6.0
τ_d2 = 8.0
τ_f = 17.0

Δ_s = 2.0
Δ_p = 10.0
Δ_a = 5.0
Δ_d1 = 0.2
Δ_d2 = 0.5
Δ_f = 2.0

# synaptic plasticity constants
τ_xi = 300.0
τ_xo = 500.0
β_i = 0.1
β_o = 0.5

# integration range in ms
tspan = [0., 2000.]
dts = 0.1
n_steps = Int32(tspan[2]/dts)
time = LinRange(tspan[1], tspan[2], n_steps)

# extrinsic input
cutoff = Int32(1000/dts)
in_start = cutoff + Int32(300/dts)
in_stop = cutoff + Int32(600/dts)
inp = zeros(n_steps,)
inp[in_start:in_stop] .= 1.0
I_ext_interp = LinearInterpolation(time, inp)

# function definitions
######################

# vector field evaluation function
function stn_gpe_str(du, u, p, t)

    # preparations
	##############

	# extract state variables
    r_s, v_s, r_p, v_p, r_a, v_a, x_a, r_d1, v_d1, x_d1, r_d2, v_d2, x_d2, r_f, v_f = u[1:15]
	r_os, r_ip_tmp, r_oa = u[[idx2, idx2+1, idx2+2]]
	r_id1, r_id2, r_if = u[[idx3, idx3+1, idx3+2]]
	r_op_tmp, r_od1, r_od2 = u[[idx4, idx4+1, idx4+2]]
	x_ip, x_op = u[[idx5, idx5+1]]

	# extract parameters
	η_s, η_p, η_a, η_d1, η_d2, η_f, k_sp, k_ps, k_pp, k_pd2, k_as, k_ap, k_ad1, k_d1a, k_d1d2, k_d1f, k_d2a, k_d2d1, k_d2f, k_fp, k_ff, α_a, α_d1, α_d2, τ_sfa, g_f, k_si, k_pi, k_ai, k_d1i, k_d2i, k_fi = p[:]

	# interpolate extrinsic input
	I_ext = I_ext_interp(t)

	# apply synaptic plasticity
	r_ip = r_ip_tmp*max((1-x_ip), 0)
	r_op = r_op_tmp*max((1-x_op), 0)

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

    # synaptic transmission
	du[idx0:idx1-1] .= k_d1 .* (r_out .- u[idx0:idx1-1])
	du[idx1:idx2-1] .= k_d2 .* (u[idx0:idx1-1] .- u[idx1:idx2-1])
	du[idx2:idx3-1] .= k_d3 .* (u[idx1:idx2-1] .- u[idx2:idx3-1])
	du[idx3:idx4-1] .= k_d4 .* (u[idx2:idx3-1][m1-m2+1:end] .- u[idx3:idx4-1])
	du[idx4:idx5-1] .= k_d5 .* (u[idx3:idx4-1][m2-m3+1:end] .- u[idx4:idx5-1])

	# synaptic plasticity
	du[idx5] = β_i*r_ip - x_ip/τ_xi
	du[idx5+1] = β_o*r_op - x_op/τ_xo

end

# definition of the initial value problem
#########################################

# state vector
N = idx5+1
u0 = zeros(N,)

# constants: dopamine level
γ = 0.0

# constants: excitabilities
η_s = 4.0
η_p = 6.0
η_a = 30.0
η_d1 = -0.5 - γ
η_d2 = -1.0 + γ
η_f = 5.0 + γ

# constants: STN input scalings
k = 1.0
k_si = 0.0 * k
k_sp = 12.0 * k

# constants: GPe-p input scalings
k_pi = 0.0 * k
k_ps = 20.0 * k
k_pp = 2.0 * k
k_pd2 = 50.0 * k

# constants: GPe-a input scalings
k_ai = 0.0 * k
k_as = 3.0 * k
k_ap = 10.0 * k
k_ad1 = 20.0 * k

# constants: MSN-D1 input scalings
k_d1i = 0.0 * k
k_d1a = 2.0 * k
k_d1d2 = 0.5 * k
k_d1f = 1.0 * k

# constants: MSN-D2 input scalings
k_d2i = 0.0 * k
k_d2a = 1.0 * k
k_d2d1 = 0.1 * k
k_d2f = 2.0 * k

# constants: FSI input scalings
k_fi = 0.0 * k
k_fp = 8.0 * k
k_ff = 1.0 * k

# constants: SFA
α_a = 0.5
α_d1 = 2.0
α_d2 = 2.0
τ_sfa = 50.0

# constants: Gap junctions
g_f = 5.0

# solve initial value problem
#############################

# parameter vector
p0 = [η_s, η_p, η_a, η_d1, η_d2, η_f, k_sp, k_ps, k_pp, k_pd2, k_as, k_ap, k_ad1, k_d1a, k_d1d2, k_d1f, k_d2a, k_d2d1, k_d2f, k_fp, k_ff, α_a, α_d1, α_d2, τ_sfa, g_f, k_si, k_pi, k_ai, k_d1i, k_d2i, k_fi]

# model definition
model = ODEProblem(stn_gpe_str, u0, tspan, p0)

# solve IVP
result = solve(model, Tsit5(), saveat=dts, reltol=1e-8, abstol=1e-8)

# plot result
target_vars = [1, 3, 5, 8, 11, 14]
display(plot(result[target_vars, cutoff:end]' .* 1e3, label=["STN" "GPe-p" "GPe-a" "MSN-D1" "MSN-D2" "FSI"]))
