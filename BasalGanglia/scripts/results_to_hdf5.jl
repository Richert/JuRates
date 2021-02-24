using HDF5, FileIO, JLD2

path = "BasalGanglia/results/stn_gpe_beta_results/"
id = "stn_gpe_beta"

params = ["tau_e", "tau_p", "tau_ampa_r", "tau_ampa_d", "tau_gabaa_r", "tau_gabaa_d", "tau_stn", "eta", "delta", "k", "eta_e", "eta_p", "k_pe", "k_ep", "k_pp"]

# go through all files in directory with identifier
files = readdir(path)
for file in files
    if startswith(file, id) && endswith(file, "params.jdl")
        @load path*file p
        i = split(file, "_")[end-1]
        @load "$path" * "$id" * "_$i" * "_fitness.jdl" f
        fn_new = "$path" * "$id" * "_$i" * ".h5"
        h5open(fn_new, "w") do file
            g = create_group(file, "p")
            for (j, n) in enumerate(params)
                g[n] = p[j]
            end
            fit = create_group(file, "f")
            fit["f"] = f
            close(file)
        end
    end
end
