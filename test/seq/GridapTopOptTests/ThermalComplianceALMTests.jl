module ThermalComplianceALMTests

using Gridap, GridapTopOpt

using ForwardDiff, Zygote
using Optim
using Krylov
using LinearMaps
using LineSearches
using PlotlyLight

function create_model()

	xmax = ymax = 1.0
	prop_Γ_N = 0.2
	prop_Γ_D = 0.2
	dom = (0,xmax,0,ymax)
	el_size = (30,30)
	order = 1

	## FE Setup
	model = CartesianDiscreteModel(dom,el_size);
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

	spaces = (U,V,V_φ,V_reg,U_reg)
	measures = (dΩ,dΓ_N,vol_D)
	el_Δ = get_el_Δ(model)
	params = (el_Δ,el_size,order)

	spaces,measures,params #el_Δ,el_size,order
end

function get_problem(η_coeff,spaces,measures,params)

	U,V,V_φ,V_reg,U_reg = spaces
	dΩ,dΓ_N,vol_D = measures
	el_Δ,el_size,order = params

	model = get_background_model(get_triangulation(spaces[1]))
	max_steps = floor(Int,minimum(el_size)/10)
	tol = 1/(5*order^2)/minimum(el_size)
	κ = 1
	vf = 0.4
	γ = 0.1
	γ_reinit = 0.5
	α_coeff = 4max_steps*γ
	α_factor = 2

	## Create FE functions
	φh = interpolate(initial_lsf(4,0.2),V_φ)

	## Interpolation and weak form
	interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
	I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

	αf = α_factor * α_coeff*maximum(el_Δ)
	af(p,q,φ) =∫(αf^2*∇(p)⋅∇(q) + p*q)dΩ;
	lf(q,φ) = ∫(q*φ)dΩ;

	a(u,v,φ) = ∫(I∘(φ)*κ*∇(u)⋅∇(v))dΩ # + ∫( 0v )dΓ_N
	l(v,φ) = ∫(v)dΓ_N

	## Optimisation functionals
	J(u,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(u))dΩ + ∫(1e-3(DH ∘ φ))dΩ;
	dJ(q,u,φ) = ∫(κ*∇(u)⋅∇(u)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;
	Vol(u,φ) = ∫(((ρ ∘ φ) - vf+0*u)/vol_D)dΩ;
	dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘Precompiling VSCodeServer...
  Progress [====================================>    ]  8/9
  ◓ VSCodeServer




 ∇(φ)))dΩ

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
		@show sum(_φ)
		Zygote.ignore() do
			φh = FEFunction(V_φ,_φ) 
			reinit!(ls_evo,φh)
			@show sum(_φ)
			println("reinit done")
		end
		φ = filter(_φ)
		u = state_map((φ))
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


# Trust region Newton-CG with Optim.jl
function f(x)
	φ_to_jc(p)[1]
end
function fg!(G,x)
	value, grad = val_and_gradient(F,x)
	copyto!(G, grad[1])
	return value
end
function hv!(Hv, x, v)
	hv = GridapTopOpt.Hpv(f,p,v) 
	println("Hv running")
	copyto!(Hv, hv)
	Hv
end
d = Optim.TwiceDifferentiableHV(f,fg!,hv!,p)
result = Optim.optimize(d, p, 
												Optim.KrylovTrustRegion(
																			initial_radius = 1.0,
																			cg_tol = 0.01,
																			#rho_upper = 0.85,
																			#eta = 0.2
																			),
												Optim.Options(g_tol = 1e-12,
																			iterations = 20,
																			store_trace = true,
																			show_trace = true,
																			extended_trace = true
																			))

val(result) = result.value
jsc = val.(result.trace)
jf = jsc




get_radius(x) = x.metadata["radius"]
radii = get_radius.(result.trace)










using NLSolversBase
using LineSearches

function g!(G,x)
	value, grad = val_and_gradient(F,x)
	copyto!(G, grad[1])
	G
end

NLSolversBase.only_fg!(fg!)

#d = Optim.TwiceDifferentiableHV(f,fg!,hv!,p)
result = Optim.optimize(f,g!, p, GradientDescent(; alphaguess = LineSearches.InitialStatic(alpha= 3.0),#,scaled = true),
                  linesearch = LineSearches.Static(),						
																),
						Optim.Options(g_tol = 1e-12,
														iterations = 20,
														store_trace = true,
														show_trace = true,
															#extended_trace = true
						))

jf2 = val.(result.trace)

plot1 = plot(x=1:length(jf),y=jf,type="scatter", mode="lines+markers") 
plot2 = plot(x=1:length(jf2),y=jf2,type="scatter", mode="lines+markers")

V_φ = spaces[3]

## Interpolation and weak form
el_Δ = params[1]
η_coeff = last(ηs)
interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

el_size = params[2]
max_steps = floor(Int,minimum(el_size)/10)
γ = 0.1
α_coeff = 4max_steps*γ
αf = 2 * α_coeff*maximum(el_Δ)
dΩ = measures[1]
af(p,q,φ) =∫(αf^2*∇(p)⋅∇(q) + p*q)dΩ;
lf(q,φ) = ∫(q*φ)dΩ;
filter = AffineFEStateMap(af,lf,V_φ,V_φ,V_φ)  

pf = result.minimizer
writevtk(get_triangulation(V_φ),"jsc",cellfields=["φ"=>FEFunction(V_φ,filter(pf)),"H(φ)"=>(H ∘ FEFunction(V_φ,filter(pf)))])

################
# Combining Plots
################

y1 = jsc
y2 = jf2
# y3 = jss[2]
# y4 = jss[3]
# y5 = jss[4]


trace1 = Config(
    x = 1:length(y1),
    y = y1,
    type = "scatter",
    mode = "lines+markers",
    name = "Newton-CG",
)

trace2 = Config(
    x = 1:length(y2),
    y = y2,
    type = "scatter",
    mode = "lines+markers",
    name = "GD"#$(γs[1])",
)

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

plot3 = Plot(
    [trace1, trace2],#, trace3, trace4, trace5],
    Config(
        title = Config(text = "Two datasets"),
        xaxis = Config(title = Config(text = "x")),
        yaxis = Config(title = Config(text = "y")),
    ),
)





end