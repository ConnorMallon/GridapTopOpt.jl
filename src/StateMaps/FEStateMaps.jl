"""
    abstract type AbstractFEStateMap

Types inheriting from this abstract type should enable the evaluation and differentiation of
the solution to an FE problem `u` that implicitly depends on an auxiliary parameter `φ`.
"""
abstract type AbstractFEStateMap end

"""
    get_state(m::AbstractFEStateMap)

Return the solution/state `u` to the FE problem.
"""
get_state(::AbstractFEStateMap) = @abstractmethod

get_state(m::Vector{<:AbstractFEStateMap}) = get_state.(m)

"""
    get_spaces(m::AbstractFEStateMap)

Return a collection of FE spaces. The first four entires should correspond to
[`get_trial_space`](@ref), [`get_test_space`](@ref), [`get_aux_space`](@ref), and
[`get_deriv_space`](@ref) unless these are overloaded for a particular implementation.
"""
get_spaces(::AbstractFEStateMap) = @abstractmethod

"""
    get_assemblers(m::AbstractFEStateMap)

Return a collection of assemblers. The first two entires should correspond to
[`get_pde_assembler`](@ref) and [`get_deriv_assembler`](@ref) unless these are
overloaded for a particular implementation.
"""
get_assemblers(::AbstractFEStateMap) = @abstractmethod

"""
    get_trial_space(m::AbstractFEStateMap)

Return trial space for FE problem.
"""
get_trial_space(m::AbstractFEStateMap) = get_spaces(m)[1]

"""
    get_test_space(m::AbstractFEStateMap)

Return test space for FE problem.
"""
get_test_space(m::AbstractFEStateMap) = get_spaces(m)[2]

"""
    get_aux_space(m::AbstractFEStateMap)

Return space for auxillary parameter.
"""
get_aux_space(m::AbstractFEStateMap) = get_spaces(m)[3]

"""
    get_deriv_space(m::AbstractFEStateMap)

Return space for derivatives.
"""
get_deriv_space(m::AbstractFEStateMap) = get_aux_space(m)

"""
    get_pde_assembler(m::AbstractFEStateMap)

Return assembler for FE problem.
"""
get_pde_assembler(m::AbstractFEStateMap) = get_assemblers(m)[1]

"""
    get_deriv_assembler(m::AbstractFEStateMap)

Return assembler for derivatives.
"""
get_deriv_assembler(m::AbstractFEStateMap) = get_assemblers(m)[2]

"""
    (φ_to_u::AbstractFEStateMap)(φh)

Evaluate the forward problem `u` given `φ`. This should compute the
FE problem.
"""
@inline (φ_to_u::AbstractFEStateMap)(φh) = forward_solve!(φ_to_u,φh)

"""
    forward_solve!(φ_to_u::AbstractFEStateMap,φh)

Evaluate the forward problem `u` given `φ`. This should compute the
FE problem.
"""
function forward_solve!(φ_to_u::AbstractFEStateMap,φh)
  @abstractmethod
end

"""
    update_incremental_state_partials!(p_to_u,res,u,p)

Update the incremental state partial `∂R/∂p` 
"""
function update_incremental_state_partials!(p_to_u,ph,diff_order::Val{2})
  U,V,V_p = p_to_u.spaces
  u̇, assem_∂R∂p, ∂R∂p_mat = p_to_u.cache.inc_state_cache
  res = get_res(p_to_u)
  uh = get_state(p_to_u)
  dv = get_fe_basis(V)
  ∂R∂p = Gridap.jacobian(p->res(uh,dv,p),ph)
  assem_∂R∂p = SparseMatrixAssembler(V_p,V)
  assemble_matrix!(∂R∂p,∂R∂p_mat,assem_∂R∂p,V_p,V)
  return ∂R∂p_mat
end

update_incremental_state_partials!(p_to_u,ph,diff_order::Val{1}) = nothing

"""
    update_adjoint_caches!(φ_to_u::AbstractFEStateMap,uh,φh)

Update the cache for the adjoint problem. This is usually a tuple
of objects.
"""
function update_adjoint_caches!(φ_to_u::AbstractFEStateMap,uh,φh)
  @abstractmethod
end

function update_adjoint_caches!(φ_to_u::AbstractFEStateMap,u::AbstractVector,φ::AbstractVector)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return update_adjoint_caches!(φ_to_u,uh,φh)
end

"""
    update_incremental_adjoint_partials(res,uh,ph,λh,spaces)

Update the incremental adjoint partials used in the second order derivative computations.
"""
function update_incremental_adjoint_partials!(p_to_u,uh,ph,λh,diff_order::Val{2})
  U,V,V_p = p_to_u.spaces
  res = get_res(p_to_u)

  if !is_cache_built(p_to_u.cache)
    build_cache!(p_to_u,ph)
  end
  _, _,   assem_∂2R∂u2, ∂2R∂u2_mat,   assem_∂2R∂u∂p,∂2R∂u∂p_mat,  assem_∂2R∂p2,∂2R∂p2_mat,  assem_∂2R∂p∂u,∂2R∂p∂u_mat = p_to_u.cache.inc_adjoint_cache

  # ∂²R / ∂u² * u̇ * λ
  ∂2R∂u2 = Gridap.hessian(uh->res(uh,λh,ph),uh) 
  assemble_matrix!(∂2R∂u2,∂2R∂u2_mat,assem_∂2R∂u2,U,U)

  # ∂/∂p (∂R/∂u * λ) * ṗ
  ∂R∂u_λ(uh,ph) = Gridap.gradient(uh->res(uh,λh,ph),uh)
  ∂2R∂u∂p = Gridap.jacobian(p->∂R∂u_λ(uh,p),ph) 
  assemble_matrix!(∂2R∂u∂p,∂2R∂u∂p_mat,assem_∂2R∂u∂p,V_p,V)

  # ∂²R / ∂p² * ṗ * λ
  ∂2R∂p2 = Gridap.hessian(ph->res(uh,λh,ph),ph)
  assemble_matrix!(∂2R∂p2,∂2R∂p2_mat,assem_∂2R∂p2,V_p,V_p)

  # ∂/∂u (∂R/∂p * λ) * ṗ
  ∂R∂p_λ(uh,ph) = Gridap.gradient(ph->res(uh,λh,ph),ph)
  ∂2R∂p∂u = Gridap.jacobian(uh->∂R∂p_λ(uh,ph),uh) 
  assemble_matrix!(∂2R∂p∂u,∂2R∂p∂u_mat,assem_∂2R∂p∂u,U,V_p)

  return ∂2R∂u2_mat, ∂2R∂u∂p_mat, ∂2R∂p2_mat, ∂2R∂p∂u_mat
end

update_incremental_adjoint_partials!(p_to_u,uh,ph,λh,diff_order::Val{1}) = nothing

"""
    adjoint_solve!(φ_to_u::AbstractFEStateMap,du::AbstractVector)

Evaluate the solution to the adjoint problem given a RHS vector `∂F∂u` denoted `du`.
This should solve the linear problem `dRduᵀ*λ = ∂F∂uᵀ`.
"""
function adjoint_solve!(φ_to_u::AbstractFEStateMap,du::AbstractVector)
  @abstractmethod
end

"""
    dRdφ(φ_to_u::AbstractFEStateMap,uh,vh,φh)

Compute the derivative with respect to `φh` of the residual R.
"""
function dRdφ(φ_to_u::AbstractFEStateMap,uh,vh,φh)
  @abstractmethod
end

function dRdφ(φ_to_u::AbstractFEStateMap,u::AbstractVector,v::AbstractVector,φ::AbstractVector)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  vh = FEFunction(get_test_space(φ_to_u),v)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return dRdφ(φ_to_u,uh,vh,φh)
end

function dRdφ(φ_to_u::AbstractFEStateMap,u::AbstractVector,v::AbstractVector,φh)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  vh = FEFunction(get_test_space(φ_to_u),v)
  return dRdφ(φ_to_u,uh,vh,φh)
end

function get_plb_cache(::AbstractFEStateMap)
  @abstractmethod
end

"""
    pullback(φ_to_u::AbstractFEStateMap,uh,φh,du;updated)

Compute `∂F∂u*dudφ` at `φh` and `uh` using the adjoint method. I.e., let

`∂F∂u*dudφ = -λᵀ*dRdφ`

and solve the adjoint problem `dRduᵀ*λ = ∂F∂uᵀ` using [`adjoint_solve!`](@ref).
"""
function pullback(φ_to_u::AbstractFEStateMap,uh,φh,du;updated=false)
  dudφ_vec, assem_deriv = get_plb_cache(φ_to_u)
  V_φ = get_deriv_space(φ_to_u)

  ## Adjoint Solve
  if !updated
    update_adjoint_caches!(φ_to_u,uh,φh)
  end

  λ  = adjoint_solve!(φ_to_u,du)
  λh = FEFunction(get_test_space(φ_to_u),λ)

  update_incremental_adjoint_partials!(φ_to_u,uh,φh,λh,get_diff_order(φ_to_u))

  ## Compute grad
  dudφ_vecdata = collect_cell_vector(V_φ,dRdφ(φ_to_u,uh,λh,φh))
  assemble_vector!(dudφ_vec,assem_deriv,dudφ_vecdata)
  rmul!(dudφ_vec, -1)

  return (NoTangent(),dudφ_vec)
end

function pullback(φ_to_u::AbstractFEStateMap,u::AbstractVector,φ::AbstractVector,du::AbstractVector;updated=false)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return pullback(φ_to_u,uh,φh,du;updated=updated)
end

"""
    rrule(φ_to_u::AbstractFEStateMap,φh)

Return the evaluation of a `AbstractFEStateMap` and a
a function for evaluating the pullback of `φ_to_u`. This enables
compatiblity with `ChainRules.jl`
"""
function ChainRulesCore.rrule(φ_to_u::AbstractFEStateMap,φh)
  u  = forward_solve!(φ_to_u,φh)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  update_adjoint_caches!(φ_to_u,uh,φh)
  return u, du -> pullback(φ_to_u,uh,φh,du;updated=true)
end

function ChainRulesCore.rrule(φ_to_u::AbstractFEStateMap,φ::AbstractVector)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return ChainRulesCore.rrule(φ_to_u,φh)
end

## Caching
mutable struct FEStateMapCache
  cache_built::Bool
  solvers::Tuple
  fwd_cache::Tuple
  adj_cache::Tuple
  plb_cache::Tuple
  inc_state_cache::Tuple
  inc_adjoint_cache::Tuple
  state_updated:: Bool
  adjoint_updated:: Bool
end

function FEStateMapCache(fwd_solver,adjoint_solver)
  FEStateMapCache(false,(fwd_solver,adjoint_solver),(),(),(),(),(),false,false)
end

is_cache_built(c::FEStateMapCache) = c.cache_built

"""
    build_cache!(::AbstractFEStateMap,φh)

Build the FEStateMapCache (see AffineFEStateMap for an example)
"""
function build_inc_cache(state_map::AbstractFEStateMap,ph,uh,adjoint_x,diff_order::Val{2})
  U,V,V_p = state_map.spaces
  res = get_res(state_map)

  println("Building incremental state cache...")

  # incremental state cache 
  u̇ = similar(get_free_dof_values(uh))
  dv = get_fe_basis(V)
  ∂R∂p = Gridap.jacobian(p->res(uh,dv,p),ph)
  assem_∂R∂p = SparseMatrixAssembler(V_p,V)
  ∂R∂p_mat = assemble_matrix(∂R∂p,assem_∂R∂p,V_p,V)
  inc_state_cache = (u̇, assem_∂R∂p, ∂R∂p_mat)

  println("Building incremental adjoint cache...")

  # incremental adjoint cache 
  λh = FEFunction(V,adjoint_x)
  λ⁻ = similar(adjoint_x)
  # ∂²R / ∂u² * u̇ * λ
  ∂2R∂u2 = Gridap.hessian(uh->res(uh,λh,ph),uh) 
  assem_∂2R∂u2 = SparseMatrixAssembler(U,U)
  ∂2R∂u2_mat = assemble_matrix(∂2R∂u2,assem_∂2R∂u2,U,U)  
  # ∂/∂p (∂R/∂u * λ) * ṗ
  ∂R∂u_λ(uh,ph) = Gridap.gradient(uh->res(uh,λh,ph),uh)
  ∂2R∂u∂p = Gridap.jacobian(p->∂R∂u_λ(uh,p),ph) 
  assem_∂2R∂u∂p = SparseMatrixAssembler(V_p,V)
  ∂2R∂u∂p_mat = assemble_matrix(∂2R∂u∂p,assem_∂2R∂u∂p,V_p,V)
  # ∂²R / ∂p² * ṗ * λ
  ∂2R∂p2 = Gridap.hessian(ph->res(uh,λh,ph),ph)
  assem_∂2R∂p2 = SparseMatrixAssembler(V_p,V_p)
  ∂2R∂p2_mat = assemble_matrix(∂2R∂p2,assem_∂2R∂p2,V_p,V_p)
  # ∂/∂u (∂R/∂p * λ) * ṗ
  ∂R∂p_λ(uh,ph) = Gridap.gradient(ph->res(uh,λh,ph),ph)
  ∂2R∂p∂u = Gridap.jacobian(uh->∂R∂p_λ(uh,ph),uh) 
  assem_∂2R∂p∂u = SparseMatrixAssembler(U,V_p)
  ∂2R∂p∂u_mat = assemble_matrix(∂2R∂p∂u,assem_∂2R∂p∂u,U,V_p)
  # incremental adjoint cotangent
  dṗ_from_u = get_free_dof_values(zero(V_p))
  inc_adjoint_cache = (λ⁻, dṗ_from_u,   assem_∂2R∂u2, ∂2R∂u2_mat,   assem_∂2R∂u∂p,∂2R∂u∂p_mat,  assem_∂2R∂p2,∂2R∂p2_mat,  assem_∂2R∂p∂u,∂2R∂p∂u_mat)
  
  println("Done building incremental state/adjoint cache.")
  
  return inc_state_cache, inc_adjoint_cache
end

build_inc_cache(state_map::AbstractFEStateMap,ph,uh,adjoint_x,diff_order::Val{1}) = ((),())

function build_cache!(::AbstractFEStateMap,φh)
  @abstractmethod
end

"""
    delete_cache!(c::FEStateMapCache)

Delete the contents of FEStateMapCache and mark for build.
"""
function delete_cache!(c::FEStateMapCache)
  c.cache_built = false
  c.fwd_cache = ()
  c.adj_cache = ()
  c.plb_cache = ()
  c.inc_state_cache = ()
  c.inc_adjoint_cache = ()
  return
end

# IO
function Base.show(io::IO,object::AbstractFEStateMap)
  print(io,"$(nameof(typeof(object)))")
end