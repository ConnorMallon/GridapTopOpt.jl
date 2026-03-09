module ThermalComplianceALMTests

using Gridap, GridapTopOpt

using ForwardDiff, Zygote
using Optim
using Krylov
using LinearMaps
using LineSearches
using PlotlyLight

function get_problem(η_coeff,α_factor)
	order = 1 
	xmax = ymax = 1.0
	prop_Γ_N = 0.2
	prop_Γ_D = 0.2
	dom = (0,xmax,0,ymax)
	el_size = (30,30)
	γ = 0.1
	γ_reinit = 0.5
	max_steps = floor(Int,order*minimum(el_size)/10)
	tol = 1/(5*order^2)/minimum(el_size)
	κ = 1
	vf = 0.4
	α_coeff = 4max_steps*γ

	## FE Setup
	model = CartesianDiscreteModel(dom,el_size);
	el_Δ = get_el_Δ(model)
	f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= ymax*prop_Γ_D + eps() ||
			x[2] >= ymax-ymax*prop_Γ_D - eps()))
	f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/2 - eps() <= x[2] <=
			ymax/2+ymax*prop_Γ_N/2 + eps())
	update_labels!(1,model,f_Γ_D,"Gamma_D")
	update_labels!(2,model,f_Γ_N,"Gamma_N")

	## Triangulations and measures
	Ω = Triangulation(model)
	Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
	dΩ = Measure(Ω,2*order)
	dΓ_N = Measure(Γ_N,2*order)
	vol_D = sum(∫(1)dΩ)

	## Spaces
	reffe_scalar = ReferenceFE(lagrangian,Float64,order)
	V = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_D"])
	U = TrialFESpace(V,0.0)
	V_φ = TestFESpace(model,reffe_scalar)
	V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
	U_reg = TrialFESpace(V_reg,0)

	## Create FE functions
	φh = interpolate(initial_lsf(4,0.2),V_φ)

	## Interpolation and weak form
	interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
	I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

	αf = α_factor * α_coeff*maximum(el_Δ)
	af(p,q,φ) =∫(αf^2*∇(p)⋅∇(q) + p*q)dΩ;
	lf(q,φ) = ∫(q*φ)dΩ;

	a(u,v,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(v))dΩ # + ∫( 0v )dΓ_N
	l(v,φ) = ∫(v)dΓ_N

	## Optimisation functionals
	J(u,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(u))dΩ + ∫(1e-3(DH ∘ φ))dΩ;
	dJ(q,u,φ) = ∫(κ*∇(u)⋅∇(u)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;
	Vol(u,φ) = ∫(((ρ ∘ φ) - vf+0*u)/vol_D)dΩ;
	dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

	## Finite difference solver and level set function
	evo = FiniteDifferenceEvolver(FirstOrderStencil(2,Float64),model,V_φ;max_steps)
	reinit = FiniteDifferenceReinitialiser(FirstOrderStencil(2,Float64),model,V_φ;tol,γ_reinit)
	ls_evo = LevelSetEvolution(evo,reinit)

	## Setup solver and FE operators
	filter = AffineFEStateMap(af,lf,V_φ,V_φ,V_φ,diff_order=2)  
	state_map = AffineFEStateMap(a,l,U,V,V_φ,diff_order=2)
	objective = GridapTopOpt.StateParamMap(J,state_map,diff_order=2)
	constraint = GridapTopOpt.StateParamMap(Vol,state_map,diff_order=2)
	#pcfs =  PDEConstrainedFunctionals(J,[Vol],state_map)

	function φ_to_jc(_φ)
		φ = filter(_φ)
		u = state_map(φ)
		j = objective(u,φ) 
		c = constraint(u,φ)
		[j+c]
	end

	pcfs = CustomPDEConstrainedFunctionals(φ_to_jc,0;state_map)

	## Optimiser
	## Hilbertian extension-regularisation problems
	α = α_coeff*maximum(el_Δ)
	a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
	vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

	pcfs,ls_evo,vel_ext,φh 

end


### Getting the problem for the AL HJ benchmark
η_coeff = 2
α_factor = 0 
pcfs,ls_evo,vel_ext,φh = get_problem(η_coeff,α_factor)

## Optimiser
i=0
iter_mod = 1
jss=Vector{Float64}[]
#cs=Float64[]
path = "/home/mallon2/Documents/GridapTopOpt.jl/results/"



# γs = [0.1,0.2,0.3,0.4]

# for γ in γs
#   global i += 1
#   js = Float64[]
# 	V_φ =  φh.fe_space
#   φh = interpolate(initial_lsf(4,0.2),V_φ)

#   optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
#     γ=γ,verbose=true,constraint_names=[],maxiter=20)
#   for (it,uh,φh) in optimiser
#     push!(js,φ_to_jc(φh.free_values)[1])
#     data = ["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh]
#     iszero(it % iter_mod) && writevtk(Ω,path*"out$it",cellfields=data)
#   end
#   it = get_history(optimiser).niter; uh = get_state(pcfs)
#   #writevtk(Ω,"tmp5",cellfields=["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
#   push!(jss,js)
# end

#p = plot(x=1:length(jss[1]),y=jss[1],type="scatter", mode="lines+markers") 
#writevtk(Ω,"tmp5",cellfields=["φ"=>φh,"H(φ)"=>(H ∘ φh),"φhf"=>φhf,"Hφf"=>H ∘ φhf,"|∇(φ)|"=>(norm ∘ ∇(φh)),"|∇(φhf)|"=>(norm ∘ ∇(φhf))])

### Getting the problem for the AL HJ benchmark
η_coeff = 5
α_factor = 2

V_φ = φh.fe_space	
#φh = interpolate(initial_lsf(4,0.2),V_φ)
p0 = φh.free_values

#function optimise(η_coeff,p)

	pcfs,_,_,_ = get_problem(η_coeff,α_factor)
	φ_to_jc = 	 pcfs.φ_to_jc
	function F(p)
		φ_to_jc(p)[1]
	end
	G(p) = Zygote.gradient(F,p)[1]
	ṗ = G(p)
	Hṗ(p,ṗ) = ForwardDiff.derivative(α -> G(p + α*ṗ), 0)
	# Hṗ(p,ṗ)

	# Test on actual optimization problems
	function f(x::Vector)
		#@show x 
			#writevtk(Ω,"ji",cellfields=["φ"=>FEFunction(V_φ,filter(x)),"H(φ)"=>(H ∘ FEFunction(V_φ,filter(x)))])

			F(x)
	end

	function fg!(G,x)
			copyto!(G, Zygote.gradient(F,x)[1])
			F(x)
	end

	function hv!(Hv, x, v)
			hv = Hṗ(x,v)
			println("Hv running")
			copyto!(Hv, hv)
			Hv
	end

	d = Optim.TwiceDifferentiableHV(f,fg!,hv!,p)
	result = Optim.optimize(d, p, Optim.KrylovTrustRegion(
																					initial_radius = 1.0,
																					cg_tol = 0.01,
																				#eta = 0.2
																		
																	),
							Optim.Options(g_tol = 1e-12,
															iterations = 3,
															store_trace = true,
																show_trace = true,
																#extended_trace = true
							))

	sum(p- result.minimizer)
	val(result) = result.value
	jsc = val.(result.trace)

	return result.minimizer,jsc

#end

jscs = Vector{Float64}[]

p=p0

for η_coeff in [5,2,1]
	global p = p
	p,jsc = optimise(η_coeff,p)
	push!(jscs,jsc)
	@show sum(p)
end








# take a step 
# smooth out the real model 

# 


# Newton methods for Topology Optimisation

# Trust region 
# Quadratic model building surrogate...
# Combined with deflation ? 
# take cheaper steps ? 
# Even space adpative smootheners ? 
# Like in higher regions
# What if we took these ALL to be hyperparameters...
# And with an agentic workfow, figured out the best routines... 
# We would HAVE to make quicker code...

# I want you to etc.. 

# we have an adaptive SCALAR which you should only 

jf = vcat(jscs...)

p = plot(x=1:length(jf),y=jf,type="scatter", mode="lines+markers") 
writevtk(Ω,"jsc",cellfields=["φu"=>φh,"φ"=>FEFunction(V_φ,filter(result.minimizer)),"H(φ)"=>(H ∘ FEFunction(V_φ,filter(result.minimizer))),"|∇(φ)|"=>(norm ∘ ∇(FEFunction(V_φ,result.minimizer)))])

#################
# Combining Plots
#################

# y1 = jsc
# y2 = jss[1]
# y3 = jss[2]
# y4 = jss[3]
# y5 = jss[4]


# trace1 = Config(
#     x = 1:length(y1),
#     y = y1,
#     type = "scatter",
#     mode = "lines+markers",
#     name = "Newton-CG",
# )

# trace2 = Config(
#     x = 1:length(y2),
#     y = y2,
#     type = "scatter",
#     mode = "lines+markers",
#     name = "$(γs[1])",
# )

# trace3 = Config(
#     x = 1:length(y3),
#     y = y3,
#     type = "scatter",
#     mode = "lines+markers",
#     name = "$(γs[2])",
# )

# trace4 = Config(
#     x = 1:length(y4),
#     y = y4,
#     type = "scatter",
#     mode = "lines+markers",
#     name = "$(γs[3])",
# )

# trace5 = Config(
#     x = 1:length(y5),
#     y = y5,
#     type = "scatter",
#     mode = "lines+markers",
#     name = "$(γs[4])",
# )

# p = Plot(
#     [trace1, trace2, trace3, trace4, trace5],
#     Config(
#         title = Config(text = "Two datasets"),
#         xaxis = Config(title = Config(text = "x")),
#         yaxis = Config(title = Config(text = "y")),
#     ),
# )





end