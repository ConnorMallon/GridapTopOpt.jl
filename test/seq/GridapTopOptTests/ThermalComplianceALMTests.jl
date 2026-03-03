module ThermalComplianceALMTests

using Gridap, GridapTopOpt

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
η_coeff = 2
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

αf = α_coeff*maximum(el_Δ)
af(p,q,φ) =∫(αf^2*∇(p)⋅∇(q) + p*q)dΩ;
lf(q,φ) = ∫(q*φ)dΩ;

a(u,v,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(v))dΩ #+ ∫(0v)dΓ_N
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

## Hilbertian extension-regularisation problems
α = α_coeff*maximum(el_Δ)
a_hilb(p,q) =∫( p*q)dΩ;
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)


function φ_to_jc(_φ)
  φ = filter(_φ)
  u = state_map(φ)
  j = objective(u,φ) 
  c = constraint(u,φ)
  [j+0.5c]
end

pcfs = CustomPDEConstrainedFunctionals(φ_to_jc,0;state_map)

## Optimiser
optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
  γ,verbose=true,constraint_names=[])

# # Do a few iterations
# vars, state = iterate(optimiser)
# vars, state = iterate(optimiser,state)

# ## Optimiser
# iter_mod = 10
# js=Float64[]
# cs=Float64[]
# path = "/home/mallon2/Documents/GridapTopOpt.jl/results/"
# optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
#   γ,verbose=true,constraint_names=[],maxiter=100)
# for (it,uh,φh) in optimiser
#   j = objective(uh,φh)
#   c = constraint(uh,φh)
#   push!(js,j)
#   push!(cs,c)
#   data = ["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh]
#   iszero(it % iter_mod) && writevtk(Ω,path*"out$it",cellfields=data)
#   #write_history(path*"/history.txt",optimiser.history)
# end
# it = get_history(optimiser).niter; uh = get_state(pcfs)
# #writevtk(Ω,path*"out$it",cellfields=["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])

# using PlotlyLight
# p = plot(x=1:length(js),y=js,type="scatter", mode="lines+markers") 
# p2 = plot(x=1:length(cs),y=cs,type="scatter", mode="lines+markers")

# # Now move to an explicit method... 
# # and lets see what happens there ......
# # using NLopt...  :: 

# # p_to_j(p) = φ_to_jc(p)[1]
# # ∇f = p->Zygote.gradient(p->p_to_j(p)[1],p)[1]
# # Hṗ(p,ṗ) =  ForwardDiff.derivative(α -> ∇f(p + α*ṗ), 0)
# # p=φh.free_values
# # p_to_j(p)
# # ∇f(p)
# # Hṗ(p,p)




# # p = φh.free_values
# # ṗ = φh.free_values 
# # ∇f = p->Zygote.gradient(p_to_j,p)[1]

# # state_map(p)
# # Zygote.gradient(p->objective(state_map(p),p),p) # update λ and u

# # Hṗ_FOR =  ForwardDiff.derivative(α -> ∇f(p + α*ṗ), 0)

# # #Affine state map Tests
# # # f(x) = 1 
# # # a(u,v,p) = ∫( p*(p+1)*∇(u)⋅∇(v) )dΩ
# # # l(v,p) = ∫( f*v )dΩ

# # a(u,v,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(v))dΩ + ∫(φ*u*v)dΓ_N
# # l(v,φ) = ∫(v*φ)dΓ_N + ∫(v*φ)dΩ

using ForwardDiff, Zygote
using Optim
using Krylov
using LinearMaps
using LineSearches


function F(p)
  u = copy(state_map(p))
  j = objective(u,p)
end

p = φh.free_values
G(p) = Zygote.gradient(F,p)[1]
ṗ = G(p)
Hṗ(p,ṗ) =  ForwardDiff.derivative(α -> G(p + α*ṗ), 0)
Hṗ(p,ṗ)

# # Test on actual optimization problems
# function f(x::Vector)
#     F(x)
# end

# function fg!(G,x)
#     copyto!(G, Zygote.gradient(F,x)[1])
#     F(x)
# end

# function hv!(Hv, x, v)
#     hv = Hṗ(x,v)
#     println("Hv running")
#     copyto!(Hv, hv)
#     Hv
# end

# d = Optim.TwiceDifferentiableHV(f,fg!,hv!,p)
# result = Optim.optimize(d, p, Optim.KrylovTrustRegion(),
#             Optim.Options(g_tol = 1e-12,
#                              iterations = 100,
#                              store_trace = true,
#                               show_trace = true,
#             ))
# sum(p- result.minimizer)


# val(result) = result.value
# jsc = val.(result.trace)

# p = plot(x=1:length(js),y=js.+cs,type="scatter", mode="lines+markers") 
# p = plot(x=1:length(jsc),y=jsc,type="scatter", mode="lines+markers") 

# writevtk(Ω,"jsc",cellfields=["φ"=>FEFunction(V_φ,result.minimizer),"H(φ)"=>(H ∘ FEFunction(V_φ,result.minimizer)),"|∇(φ)|"=>(norm ∘ ∇(FEFunction(V_φ,result.minimizer)))])





function Hṗ_int(Hṗ_p_v)
  Hpₕ = FEFunction(V_φ,Hṗ_p_v)
  a(u,v) = ∫(u*v)dΩ
  l(v) = ∫(v*Hpₕ)dΩ
  op = AffineFEOperator(a,l,V_φ,V_φ)
  get_vector(op)
end

A(p) = LinearMap((v)->Hṗ_int(Hṗ(p,v)),length(p),length(p))

using Krylov
p0 = φh.free_values

a(u,v) = ∫( 0.1*u*v + 0.1* ∇(u)⋅∇(v) )dΩ
assem = SparseMatrixAssembler(V_φ,V_φ)
S = assemble_matrix(a,assem,V_φ,V_φ)

function b(g)
  gdh = FEFunction(V_φ,g)
  l(v) = ∫( v*gdh )dΩ
  assemble_vector(l,V_φ )
end

g0 = G(p0)
x,stats = minares(A(p0)+S,b(g0),verbose=1,itmax=300)
x



# Test on actual optimization problems
function f(x::Vector)
    F(x)
end

function fg!(G,x)
    g = Zygote.gradient(F,x)[1]
    #@show g
    H⁻¹g,stats = minares(A(x)+S,b(g),verbose=1,itmax=300)
    copyto!(G, H⁻¹g)
    F(x)
end


# #algo = Newton(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.MoreThuente())
# algo = GradientDescent(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.Static())


# result = Optim.optimize(f, fg!, p0, algo
# ),
#             Optim.Options(g_tol = 1e-12,
#                              iterations = 10,
#                              store_trace = true,
#                               show_trace = true,
#             )


jsc = []
function my_gd(f,fg!,p0;maxiter=10)
  p = copy(p0)
  for i in 1:maxiter
  f, g = Zygote.withgradient(f,p)[1]
    push!(js,f)
    println("Iter $i: f = $f")
    H⁻¹g,stats = minares(A(p)+S,b(g),verbose=1,itmax=300)
    p -= H⁻¹g
  end
  p
end





#p = plot(x=1:length(js),y=js.+cs,type="scatter", mode="lines+markers") 
p = plot(x=1:length(jsc),y=jsc,type="scatter", mode="lines+markers") 

writevtk(Ω,"jsc",cellfields=["φ"=>FEFunction(V_φ,result.minimizer),"H(φ)"=>(H ∘ FEFunction(V_φ,result.minimizer)),"|∇(φ)|"=>(norm ∘ ∇(FEFunction(V_φ,result.minimizer)))])





# Can we solve the problem only on the surface ?
end