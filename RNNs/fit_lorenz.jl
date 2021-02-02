using MultivariateStats,FileIO,Plots,MLBase

# load data
data_lorenz = load("qif_exp_lorenz_chaotic.jdl", "data")
data_sl = load("qif_exp_stuartlandau_chaotic.jdl", "data")
#data_sl = hcat(data[i] for i in range(1,length(data),step=1))

# create full data matrix and target vector
X_tmp = vcat([data_lorenz; data_sl])
n = length(X_tmp)
X = zeros((n, length(X_tmp[1])))
for i=1:n
    X[i, :] .= X_tmp[i]
end
y_lorenz = zeros(length(data_lorenz),) .- 1.0
y_sl = zeros(length(data_sl),) .+ 1.0
y = vcat([y_lorenz; y_sl])

# define fitting function
function train_readout(idx)
    α = 0.1
    X_tmp = X[idx, :]
    y_tmp = y[idx]
    sol = ridge(X_tmp, y_tmp, α)
    return sol
end

# define testing function
function test_readout(model, idx)
    θ, μ = model[1:end-1], model[end]
    X_tmp = X[idx, :]
    y_tmp = y[idx]
    prediction = X_tmp * θ .+ μ
    diff = y_tmp .- prediction
    rmse = sqrt((diff' * diff)/length(diff))
    return rmse
end

# perform cross-validation on prediction performance of readout
scores = cross_validate(train_readout, test_readout, n, Kfold(n, 10))
(m, s) = mean_and_std(scores)
display("rmse mean = $m, rmse std = $s")
