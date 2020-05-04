using HDF5, FileIO, JLD2

path = "BasalGanglia/results/stn_gpe_ev_opt_results_final/"
id = "stn_gpe_ev_opt"

params = ["eta_e", "eta_p", "eta_a", "k_ee", "k_pe", "k_ae", "k_ep", "k_pp", "k_ap", "k_pa", "k_aa", "k_ps", "k_as", "delta_e", "delta_p", "delta_a"]

# go through all files in directory with identifier
files = readdir(path)
for file in files
    if occursin(id, file) && occursin(".jdl", file)
        if occursin("params", file)
            @load path*file p
            i = split(file, "_")[end-1]
            fn_new = "$path" * "$id" * "_$i" * "_params.h5"
            h5open(fn_new, "w") do file
                g = g_create(file, "p") # create a group
                for (j, n) in enumerate(params)
                    g[n] = p[j]
                end
                attrs(g)["description"] = "parameter fits" # an attribute
            end
        else
            @load path*file f
            i = split(file, "_")[end-1]
            fn_new = "$path" * "$id" * "_$i" * "_fitness.h5"
            h5write(fn_new, "f", f)
        end
    end
end
