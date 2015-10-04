1+2

linear_transit_ephemeris(transit_num, param::Vector) = param[1]+param[2]*transit_num

linear_transit_ephemeris(3,[1,2])
linear_transit_ephemeris([1.,2.,3.],[1.,2.])
linear_transit_ephemeris(1.0:3.0,[1.,2.])

# Generic function to calculate chi squared for a given model and data
# Inputs:
#   param:  Vector of model parameters
#   model:  Function taking x values and input parameters and returning model predictions
#   x:      Data for input to model
#   y:      Measured data for comparison to model predictions
#   sigma:  Uncertainties in measurements
# Output: chi squared of model for given parameters and data
function chisq_model_vs_data(param::Vector, model::Function, x::Vector, y::Vector, sigma::Vector)
  @assert( length(x) == length(y) == length(sigma) > 0)
  chisq = zero(eltype(param))
  for i in 1:length(x)
    predict = model(x[i],param)
    chisq += ((predict-y[i])/sigma[i])^2
  end
  chisq
end


P_b_true = 5.729
 t0_b_true = 781.99
 param_true = [t0_b_true,P_b_true]
 const sim_trid_list_b = collect(linspace(-125,129,255))
 const true_tt_list_b = linear_transit_ephemeris(sim_trid_list_b,param_true)
 const sigma_tt_b = 0.005*ones(length(true_tt_list_b))
 const sim_tt_list_b = true_tt_list_b + sigma_tt_b.*randn(length(true_tt_list_b))

@time chisq_model_vs_data(param_true,linear_transit_ephemeris,sim_trid_list_b,sim_tt_list_b,sigma_tt_b)
param_guess = param_true + 1e-5*randn(length(param_true))
@time chisq_model_vs_data(param_guess,linear_transit_ephemeris,sim_trid_list_b,sim_tt_list_b,sigma_tt_b)

chisq_linear_b(param::Vector) = chisq_model_vs_data(param,linear_transit_ephemeris,sim_trid_list_b,sim_tt_list_b,sigma_tt_b)
chisq_linear_b(param_true)

P_b_guess = P_b_true+0.000001*randn()
 t0_b_guess = t0_b_true+0.00001*randn()
 pl_b_guess = [t0_b_guess,P_b_guess]
 chisq_linear_b(pl_b_guess)

# Pkg.add("Optim")
using Optim
fit_b_output = optimize(chisq_linear_b,pl_b_guess)

fit_b_output.minimum-[t0_b_true,P_b_true]
fit_b_output.f_minimum

# What if you are used to writing "vectorized" expressions like in Python/IDL/R?
function chisq_model_vs_data_python_lovers(param::Vector, model::Function, x::Vector, y::Vector, sigma::Vector)
  chisq = sum(((model(x,param)-y)./sigma).^2)
end

chisq_linear_b(param::Vector) = chisq_model_vs_data_python_lovers(param,linear_transit_ephemeris,sim_trid_list_b,sim_tt_list_b,sigma_tt_b)

chisq_model_vs_data_python_lovers(param_true,linear_transit_ephemeris,sim_trid_list_b,sim_tt_list_b,sigma_tt_b)
chisq_model_vs_data_python_lovers(param_guess,linear_transit_ephemeris,sim_trid_list_b,sim_tt_list_b,sigma_tt_b)

#Pkg.add("ForwardDiff")
using ForwardDiff
grad_chisq = ForwardDiff.gradient(chisq_linear_b)
grad_chisq(param_true)

fit_b_output = optimize(chisq_linear_b,pl_b_guess,method = :bfgs, autodiff= true)

#Pkg.add("PyCall")
using PyCall
@pyimport numpy.random as nr
nr.rand(3,4)


using PyPlot
pplt.clf()
resid = sim_tt_list_b.-linear_transit_ephemeris(sim_trid_list_b,fit_b_output.minimum)
 pplt = PyPlot
 pplt.errorbar(sim_tt_list_b, resid, yerr=sigma_tt_b, fmt="o")
 pplt.title("Sample graph")
 pplt.figure(1)



gen_random_param(x) = Float64[t0_b_true+0.00001*randn(),P_b_true+0.00001*randn()]
param_list = [gen_random_param(i) for i in 1:1000]
P_list = Float64[param_list[i][2] for i in 1:length(param_list) ]
chisq_list = map(chisq_linear_b, param_list)

pplt.clf()
 pplt.plot(P_list,chisq_list,"ro")
 pplt.xlabel("Period")
 pplt.ylabel("\chi^2")
 pplt.figure(1)


# Begin Untested

workspace()
addprocs(4)

@everywhere linear_transit_ephemeris(transit_num, param::Vector) = param[1]+param[2]*transit_num
@everywhere  sim_trid_list_b = collect(linspace(-125,129,255))
@everywhere const P_b_true = 5.729
@everywhere const t0_b_true = 781.99
@everywhere param_true = [t0_b_true,P_b_true]
@everywhere const  true_tt_list_b = linear_transit_ephemeris(sim_trid_list_b,param_true)
@everywhere const  sigma_tt_b = 0.005*ones(length(true_tt_list_b))
@everywhere const  sim_tt_list_b = true_tt_list_b + sigma_tt_b.*randn(length(true_tt_list_b))

@everywhere gen_random_param(x) = Float64[t0_b_true+0.001*randn(),P_b_true+0.001*randn()]
@everywhere chisq_model_vs_data_python_lovers(param::Vector, model::Function, x::Vector, y::Vector, sigma::Vector) =  chisq = sum(((model(x,param)-y)./sigma).^2)
@everywhere chisq_linear_b(param::Vector) = chisq_model_vs_data_python_lovers(param,linear_transit_ephemeris,sim_trid_list_b,sim_tt_list_b,sigma_tt_b)

chisq_list_parallel = pmap(chisq_linear_b, param_list)

