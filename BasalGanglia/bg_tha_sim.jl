using DifferentialEquations, Plots, Interpolations

# constant definitions and initializations
##########################################

# global constants
c = 1/π^2

# membrane time constants
τ_s = 12.2
τ_p1 = 14.7
τ_p2 = 20.0
τ_a = 23.4
τ_d1 = 7.3
τ_d2 = 6.2
τ_f = 17.2
τ_g = 13.0
τ_t = 12.0
τ_i = 20.0

# neural heterogeneities
Δ_s = 0.8
Δ_p1 = 6.5
Δ_p2 = 3.0
Δ_a = 2.6
Δ_d1 = 2.0
Δ_d2 = 4.0
Δ_f = 1.5
Δ_g = 4.0
Δ_t = 1.0
Δ_i = 1.5

# gap junctions
g_f = 0.5

# state vector
N = 20
u0 = zeros(N,)

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
function bg_tha(du, u, p, t)

    # preparations
	##############

	# extract state variables
    r_s, v_s, r_p1, v_p1, r_p2, v_p2, r_a, v_a, r_d1, v_d1, r_d2, v_d2, r_f, v_f, r_g, v_g, r_t, v_t, r_i, v_i= u[:]

	# extract parameters
	η_s, η_p1, η_p2, η_a, η_d1, η_d2, η_f, η_g, η_t, η_i, k_se, k_sp1, k_st, k_p1e, k_p1s, k_p1t, k_p1p1, k_p1p2, k_p1d2, k_p2e, k_p2s, k_p2t, k_p2p1, k_p2p2, k_p2d1, k_ae, k_as, k_at, k_ap1, k_ad1, k_d1e, k_d1t, k_d1a, k_d1f, k_d2e, k_d2t, k_d2a, k_d2f, k_fe, k_fs, k_ft, k_fp1, k_ff, k_ge, k_gs, k_gt, k_gp1, k_gd1, k_te, k_tt, k_tp2, k_tg, k_ti, k_ie, k_it, k_ii = p[:]

	# interpolate extrinsic input
	I_ext = I_ext_interp(t)

	# define synaptic inputs
	s_s = k_se*I_ext + k_st*r_t - k_sp1*r_p1
 	s_p1 = k_p1e*I_ext + k_p1s*r_s + k_p1t*r_t - k_p1p1*r_p1 - k_p1p2*r_p2 - k_p1d2*r_d2
	s_p2 = k_p2e*I_ext + k_p2s*r_s + k_p2t*r_t - k_p2p1*r_p1 - k_p2p2*r_p2 - k_p2d1*r_d1
	s_a = k_ae*I_ext + k_as*r_s + k_at*r_t - k_ap1*r_p1 - k_ad1*r_d1
	s_d1 = k_d1e*I_ext + k_d1t*r_t - k_d1a*r_a - k_d1f*r_f
	s_d2 = k_d2e*I_ext + k_d2t*r_t - k_d2a*r_a - k_d2f*r_f
	s_f = k_fe*I_ext + k_fs*r_s + k_ft*r_t - k_fp1*r_p1 - k_ff*r_f
	s_g = k_ge*I_ext + k_gs*r_s + k_gt*r_t - k_gp1*r_p1 - k_gd1*r_d1
	s_t = k_te*I_ext + k_tt*r_t - k_tp2*r_p2 - k_tg*r_g - k_ti*r_i
	s_i = k_ie*I_ext + k_it*r_t - k_ii*r_i

    # populations
    #############

    # STN
    du[1] = (Δ_s/(π*τ_s) + 2.0*r_s*v_s) * (c/τ_s)
    du[2] = (v_s^2 + (η_s*c + s_s*τ_s - (τ_s*π*r_s)^2)/c^2) * (c/τ_s)

    # GPe-p I
    du[3] = (Δ_p1/(π*τ_p1) + 2.0*r_p1*v_p1) * (c/τ_p1)
    du[4] = (v_p1^2 + (η_p1*c + s_p1*τ_p1 - (τ_p1*π*r_p1)^2)/c^2) * (c/τ_p1)

	# GPe-p II
    du[5] = (Δ_p2/(π*τ_p2) + 2.0*r_p2*v_p2) * (c/τ_p2)
    du[6] = (v_p2^2 + (η_p2*c + s_p2*τ_p2 - (τ_p2*π*r_p2)^2)/c^2) * (c/τ_p2)

    # GPe-a
    du[7] = (Δ_a/(π*τ_a) + 2.0*r_a*v_a) * (c/τ_a)
    du[8] = (v_a^2 + (η_a*c + s_a*τ_a - (τ_a*π*r_a)^2)/c^2) * (c/τ_a)

	# MSN-D1
	du[9] = (Δ_d1/(π*τ_d1) + 2.0*r_d1*v_d1) * (c/τ_d1)
    du[10] = (v_d1^2 + (η_d1*c + s_d1*τ_d1 - (τ_d1*π*r_d1)^2)/c^2) * (c/τ_d1)

	# MSN-D2
	du[11] = (Δ_d2/(π*τ_d2) + 2.0*r_d2*v_d2) * (c/τ_d2)
    du[12] = (v_d2^2 + (η_d2*c + s_d2*τ_d2 - (τ_d2*π*r_d2)^2)/c^2) * (c/τ_d2)

	# FSI
	du[13] = (Δ_f/(π*τ_f) + r_f*(2*v_f-g_f)) * (c/τ_f)
    du[14] = (v_f^2 + (η_f*c + s_f*τ_f - (τ_f*π*r_f)^2)/c^2) * (c/τ_f)

	# GPi/SNr
	du[15] = (Δ_g/(π*τ_g) + 2.0*r_g*v_g) * (c/τ_g)
    du[16] = (v_g^2 + (η_g*c + s_g*τ_g - (τ_g*π*r_g)^2)/c^2) * (c/τ_g)

	# Tha (projection neurons)
	du[17] = (Δ_t/(π*τ_t) + 2.0*r_t*v_t) * (c/τ_t)
    du[18] = (v_t^2 + (η_t*c + s_t*τ_t - (τ_t*π*r_t)^2)/c^2) * (c/τ_t)

	# Tha (interneurons)
	du[19] = (Δ_i/(π*τ_i) + 2.0*r_i*v_i) * (c/τ_i)
    du[20] = (v_i^2 + (η_i*c + s_i*τ_i - (τ_i*π*r_i)^2)/c^2) * (c/τ_i)

end

# remaining model constants definition
#######################################

# constants: excitabilities
η_s = 4.0
η_p1 = 16.0
η_p2 = 4.0
η_a = 6.0
η_d1 = -4.0
η_d2 = -8.0
η_f = 4.0
η_g = 12.0
η_t = 12.0
η_i = -10.0

# constants: STN input scalings
k_se = 0.0
k_sp1 = 10.0
k_st = 10.0

# constants: GPe-p I input scalings
k_p1e = 0.0
k_p1s = 30.0
k_p1t = 30.0
k_p1p1 = 10.0
k_p1p2 = 5.0
k_p1d2 = 80.0

# constants: GPe-p II input scalings
k_p2e = 0.0
k_p2s = 10.0
k_p2t = 30.0
k_p2p1 = 20.0
k_p2p2 = 5.0
k_p2d1 = 100.0

# constants: GPe-a input scalings
k_ae = 0.0
k_as = 5.0
k_at = 5.0
k_ap1 = 20.0
k_ad1 = 100.0

# constants: MSN-D1 input scalings
k_d1e = 0.0
k_d1t = 5.0
k_d1a = 50.0
k_d1f = 50.0

# constants: MSN-D2 input scalings
k_d2e = 0.0
k_d2t = 7.0
k_d2a = 50.0
k_d2f = 50.0

# constants: FSI input scalings
k_fe = 0.0
k_fs = 5.0
k_ft = 10.0
k_fp1 = 10.0
k_ff = 5.0

# constants: GPi/SNr input scalings
k_ge = 0.0
k_gs = 10.0
k_gt = 30.0
k_gp1 = 50.0
k_gd1 = 100.0

# constants: Tha projection neuron inputs scalings
k_te = 0.0
k_tt = 5.0
k_tp2 = 20.0
k_tg = 50.0
k_ti = 10.0

# constants: Tha interneuron inputs scalings
k_ie = 0.0
k_it = 10.0
k_ii = 5.0

# solve initial value problem
#############################

# parameter vector
p0 = [η_s, η_p1, η_p2, η_a, η_d1, η_d2, η_f, η_g, η_t, η_i, k_se, k_sp1, k_st, k_p1e, k_p1s, k_p1t, k_p1p1, k_p1p2, k_p1d2, k_p2e, k_p2s, k_p2t, k_p2p1, k_p2p2, k_p2d1, k_ae, k_as, k_at, k_ap1, k_ad1, k_d1e, k_d1t, k_d1a, k_d1f, k_d2e, k_d2t, k_d2a, k_d2f, k_fe, k_fs, k_ft, k_fp1, k_ff, k_ge, k_gs, k_gt, k_gp1, k_gd1, k_te, k_tt, k_tp2, k_tg, k_ti, k_ie, k_it, k_ii]

# model definition
model = ODEProblem(bg_tha, u0, tspan, p0)

# solve IVP
sol = solve(model, Tsit5(), saveat=dts, reltol=1e-8, abstol=1e-8)

# plot result
stn_vars = [1]
gp_vars = [3, 5, 7, 15]
str_vars = [9, 11, 13]
tha_vars = [17, 19]

p1 = plot(sol[stn_vars, cutoff:end]' .* 1e3, title="STN rates")
p2 = plot(sol[gp_vars, cutoff:end]' .* 1e3, label=["GPe-p1" "GPe-p2" "GPe-a" "GPi"], title="GP rates")
p3 = plot(sol[str_vars, cutoff:end]' .* 1e3, label=["MSN-D1" "MSN-D2" "FSI"], title="STR rates")
p4 = plot(sol[tha_vars, cutoff:end]' .* 1e3, label=["PN" "IN"], title="THA rates")

display(plot(p1, p2, p3, p4, layout=(4,1), size=(600, 600)))
