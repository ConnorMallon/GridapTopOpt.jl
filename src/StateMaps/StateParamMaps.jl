# Abstract type is only needed for compat with staggered state maps. This
#  type will be deprecated in a future release.
abstract type AbstractStateParamMap end

"""
    struct StateParamMap{A,B,C,D} <: AbstractStateParamMap

A wrapper to handle partial differentation of a function F
of a specific form (see below) in a `ChainRules.jl` compatible way with caching.

# Assumptions

We assume that we have a function F of the following form:

`F(u,φ) = ∫(f(u,φ))dΩ₁ + ∫(g(u,φ))dΩ₂ + ...,`.

where `u` and `φ` are each expected to inherit from `Union{FEFunction,MultiFieldFEFunction}`
or the GridapDistributed equivalent.
"""
struct StateParamMap{A,B,C,D,N} <: AbstractStateParamMap
  F       :: A
  spaces  :: B
  assems  :: C
  cache  :: D

  """
      StateParamMap(F,U::FESpace,V_φ::FESpace,
      assem_U::Assembler,assem_deriv::Assembler)

  Create an instance of `StateParamMap`.

  Use the optional argument `∂F∂u` and/or `∂F∂φ`  to specify the directional derivative of
  F(u,φ) with respect to the field u in the direction q as ∂F∂u(q,u,φ) and/or with respect
  to the field φ in the direction q as ∂F∂φ(q,u,φ).

  Optional arguments `∂u_ad_type` and `∂φ_ad_type` specify the approach for AD for multifield
  problems (either :split or :monolithic). For SingleField FE problems, this does nothing. Description of options
  can be found in Gridap.MultiField.
  """
  function StateParamMap(
    F,U::FESpace,V_φ::FESpace,
    assem_U::Assembler,assem_deriv::Assembler;
    ∂u_ad_type::Symbol=:split,
    ∂φ_ad_type::Symbol=:monolithic,
    ∂F∂u::Function = (q,u,φ) -> __gradient(x->F(x,φ),u;ad_type=∂u_ad_type),
    ∂F∂φ::Function = (q,u,φ) -> __gradient(x->F(u,x),φ;ad_type=∂φ_ad_type),
    diff_order::Int = 1
  )
    ## Dev note (commit fd65d0a):
    # In the past we used the following code to allocate vectors for the derivatives.
    # This was required because we needed these to be RHS vectors for VelocityExtension
    # problem. As of v0.3.0 (commmit fd65d0a), this is no longer required because VelocityExtension
    # expects dF to be a vector of DOFs. This is then mapped onto an appropriate RHS vector
    # using `_interpolate_onto_rhs!`.
    #
    # In `u_to_j_pullback` below, we do in-place assembly via `assemble_vector!` on these
    # allocated vectors. This is a bit naughty but works!
    #######
    # φ₀, u₀ = interpolate(x->-sqrt((x[1]-1/2)^2+(x[2]-1/2)^2)+0.2,V_φ), zero(U)
    # ∂j∂u_vecdata = collect_cell_vector(U,_∂F∂u(get_fe_basis(U),u₀,φ₀))
    # ∂j∂φ_vecdata = collect_cell_vector(V_φ,∇(F,[u₀,φ₀],2))
    # ∂j∂u_vec = allocate_vector(assem_U,∂j∂u_vecdata)
    # ∂j∂φ_vec = allocate_vector(assem_deriv,∂j∂φ_vecdata)
    #######

    assems = (assem_U,assem_deriv)
    spaces = (U,V_φ)
    cache = StateParamMapCache(∂F∂u,∂F∂φ)

    A, B, C, D = typeof(F), typeof(spaces), typeof(assems), typeof(cache)
    if diff_order == 1
      return new{A,B,C,D,1}(
        F,spaces,assems,cache
      )
    elseif diff_order == 2
      return new{A,B,C,D,2}(
        F,spaces,assems,cache
      )
    else
      error("Unsupported diff_order = $diff_order. Expected 1 or 2.")
    end
  end
end

build_inc_obj_cache(F,uh,ph,spaces,diff_order::Val{1}) = ()

function build_inc_obj_cache(F,uh,ph,spaces,diff_order::Val{2})
  U,V_p = spaces
  
  println("Building incremental objective cache for second derivatives. This may take some time...")

  # ∂²J / ∂u² * u̇
  ∂2J∂u2 = Gridap.hessian(uh->F(uh,ph),uh)
  assem_∂2J∂u2 = SparseMatrixAssembler(U,U)
  ∂2J∂u2_mat = assemble_matrix(∂2J∂u2,assem_∂2J∂u2,U,U)

  # ∂/∂p (∂J/∂u ) * ṗ
  ∂J∂u(uh,ph) = Gridap.gradient(uh->F(uh,ph),uh)
  ∂2J∂u∂p = Gridap.jacobian(p->∂J∂u(uh,p),ph)
  assem_∂2J∂u∂p = SparseMatrixAssembler(V_p,U)
  ∂2J∂u∂p_mat = assemble_matrix(∂2J∂u∂p,assem_∂2J∂u∂p,V_p,U)

  # ∂²J / ∂p² * ṗ
  ∂2J∂p2 = Gridap.hessian(p->F(uh,p),ph)
  assem_∂2J∂p2 = SparseMatrixAssembler(V_p,V_p)
  ∂2J∂p2_mat = assemble_matrix(∂2J∂p2,assem_∂2J∂p2,V_p,V_p)

  # ∂/∂u (∂J / ∂p) * u̇
  ∂J∂p(uh,ph) = Gridap.gradient(p->F(uh,p),ph)
  ∂2J∂p∂u = Gridap.jacobian(uh->∂J∂p(uh,ph),uh)
  assem_∂2J∂p∂u = SparseMatrixAssembler(U,V_p)
  ∂2J∂p∂u_mat = assemble_matrix(∂2J∂p∂u,assem_∂2J∂p∂u,U,V_p)

  println("Done building incremental objective cache.")

  dṗ_from_j = get_free_dof_values(zero(V_p))
  du̇_from_j = get_free_dof_values(zero(U))

  dṗ_from_j, du̇_from_j, assem_∂2J∂u2, ∂2J∂u2_mat,   assem_∂2J∂u∂p, ∂2J∂u∂p_mat,   assem_∂2J∂p2, ∂2J∂p2_mat,   assem_∂2J∂p∂u, ∂2J∂p∂u_mat
end

function update_inc_obj_cache!(inc_obj_cache,F,uh,ph,spaces,diff_order::Val{1})
  return inc_obj_cache
end

function update_inc_obj_cache!(inc_obj_cache,F,uh,ph,spaces,diff_order::Val{2})
  U,V_p = spaces 
  dṗ_from_j, du̇_from_j, assem_∂2J∂u2, ∂2J∂u2_mat,   assem_∂2J∂u∂p, ∂2J∂u∂p_mat,   assem_∂2J∂p2, ∂2J∂p2_mat,   assem_∂2J∂p∂u, ∂2J∂p∂u_mat = inc_obj_cache

  ∂2J∂u2 = Gridap.hessian(uh->F(uh,ph),uh)
  assemble_matrix!(∂2J∂u2,∂2J∂u2_mat,assem_∂2J∂u2,U,U)

  ∂J∂u(uh,ph) = Gridap.gradient(uh->F(uh,ph),uh)
  ∂2J∂u∂p = Gridap.jacobian(p->∂J∂u(uh,p),ph)
  assemble_matrix!(∂2J∂u∂p,∂2J∂u∂p_mat,assem_∂2J∂u∂p,V_p,U)

  ∂2J∂p2 = Gridap.hessian(p->F(uh,p),ph)
  assemble_matrix!(∂2J∂p2,∂2J∂p2_mat,assem_∂2J∂p2,V_p,V_p)

  ∂J∂p(uh,ph) = Gridap.gradient(p->F(uh,p),ph)
  ∂2J∂p∂u = Gridap.jacobian(uh->∂J∂p(uh,ph),uh)
  assemble_matrix!(∂2J∂p∂u,∂2J∂p∂u_mat,assem_∂2J∂p∂u,U,V_p)

  return inc_obj_cache
end

function get_∂F∂φ_vec(u_to_j::StateParamMap)
  u_to_j.caches[2]
end
get_state(m::StateParamMap) = FEFunction(m.spaces[1], m.caches[5])
get_parameter(m::StateParamMap) = FEFunction(m.spaces[2], m.caches[6])
get_diff_order(m::StateParamMap{A,B,C,D,N}) where {A,B,C,D,N} = Val(N)

function StateParamMap(F::Function,φ_to_u::AbstractFEStateMap;kwargs...)
  U = get_trial_space(φ_to_u)
  V_φ = get_aux_space(φ_to_u)
  assem_deriv = get_deriv_assembler(φ_to_u)
  assem_U = get_pde_assembler(φ_to_u)
  StateParamMap(F,U,V_φ,assem_U,assem_deriv;kwargs...)
end

"""
    (u_to_j::StateParamMap)(uh,φh)

Evaluate the `StateParamMap` at parameters `uh` and `φh`.
"""
function (u_to_j::StateParamMap)(uh::FEFunction,φh::FEFunction)
  diff_order = get_diff_order(u_to_j)

  if !is_cache_built(u_to_j.cache)
    build_cache!(u_to_j,uh,φh)
  end
  u_to_j.cache.fwd_ran = true # (running fwd here)
  u_to_j.cache.bwd_ran = false # (bwd to be set to true in the pullback)
  u_to_j(uh,φh,diff_order)
end

(u_to_j::AbstractStateParamMap)(uh,φh,diff_order::Val{1}) = sum(u_to_j.F(uh,φh))

function (u_to_j::StateParamMap)(uh::FEFunction,φh::FEFunction,diff_order::Val{2})
  u, p, j = u_to_j.cache.fwd_cache
  copyto!(u, get_free_dof_values(uh))
  copyto!(p, get_free_dof_values(φh))
  j[] = sum(u_to_j.F(uh, φh))
end

function (u_to_j::StateParamMap)(u::AbstractVector,φ::AbstractVector)
  U,V_φ = u_to_j.spaces
  uh = FEFunction(U,u)
  φh = FEFunction(V_φ,φ)
  return u_to_j(uh,φh)
end

"""
    ChainRulesCore.rrule(u_to_j::StateParamMap,uh,φh)

Return the evaluation of a `StateParamMap` and a
a function for evaluating the pullback of `u_to_j`. This enables
compatiblity with `ChainRules.jl`
"""

function pullback(u_to_j::StateParamMap,uh,φh,dj)
  F = u_to_j.F
  U,V_φ = u_to_j.spaces
  assem_U,assem_deriv = u_to_j.assems
  ∂j∂u_vec,∂j∂φ_vec,∂F∂u,∂F∂φ = u_to_j.cache.plb_cache

  ## Compute ∂F/∂uh(uh,φh) and ∂F/∂φh(uh,φh)
  ∂j∂u = ∂F∂u(get_fe_basis(U),uh,φh)
  ∂j∂u_vecdata = collect_cell_vector(U,∂j∂u)
  assemble_vector!(∂j∂u_vec,assem_U,∂j∂u_vecdata)
  ∂j∂φ = ∂F∂φ(get_fe_basis(V_φ),uh,φh)
  ∂j∂φ_vecdata = collect_cell_vector(V_φ,∂j∂φ)
  assemble_vector!(∂j∂φ_vec,assem_deriv,∂j∂φ_vecdata)
  ∂j∂u_vec .*= dj
  ∂j∂φ_vec .*= dj
  update_inc_obj_cache!(u_to_j.cache.inc_obj_cache,u_to_j.F,uh,φh,u_to_j.spaces,get_diff_order(u_to_j))
  u_to_j.cache.bwd_ran = true 
  (  NoTangent(), ∂j∂u_vec, ∂j∂φ_vec )
end

function pullback(u_to_j::StateParamMap,u::AbstractVector,φ::AbstractVector,dj)
  U,V_φ = u_to_j.spaces
  uh = FEFunction(U,u)
  φh = FEFunction(V_φ,φ)
  return pullback(u_to_j,uh,φh,dj)
end

function ChainRulesCore.rrule(u_to_j::StateParamMap,uh,φh)
  return u_to_j(uh,φh), dj -> pullback(u_to_j,uh,φh,dj)
end

"""
    rrule(u_to_j::StateParamMap,uh,φh)

Return the evaluation of a `StateParamMap` and a
function for evaluating the pullback of `u_to_j`. This enables
compatibility with `ChainRules.jl`
"""
function rrule(u_to_j::StateParamMap,uh,φh)
  ChainRulesCore.rrule(u_to_j,uh,φh)
end

function rrule(u_to_j::StateParamMap,u::AbstractVector,φ::AbstractVector)
  ChainRulesCore.rrule(u_to_j,u,φ)
end

# IO
function Base.show(io::IO,object::AbstractStateParamMap)
  print(io,"$(nameof(typeof(object)))")
end

# Backwards compat
function StateParamIntegrandWithMeasure(args...)
  error(
    """
    As of v0.4.0, StateParamIntegrandWithMeasure was deprecated in favour of StateParamMap.
    """
  )
end
function StateParamMap(
    F,U::FESpace,V_φ::FESpace,U_reg,assem_U::Assembler,assem_deriv::Assembler;kwargs...)
  error(_msg_v0_3_0(StateParamMap))
end

mutable struct StateParamMapCache
  fwd_cache::Tuple
  plb_cache::Tuple
  inc_obj_cache::Tuple
  cache_built::Bool
  fwd_ran:: Bool
  bwd_ran:: Bool
end

is_cache_built(c::StateParamMapCache) = c.cache_built

function StateParamMapCache(∂F∂u,∂F∂φ)
  plb_cache = (nothing, nothing, ∂F∂u, ∂F∂φ)  
  StateParamMapCache((),plb_cache,(),false,false,false)
end

function build_cache!(u_to_j::StateParamMap,uh,φh)
  _,_,∂F∂u,∂F∂φ = u_to_j.cache.plb_cache
  U,V_φ = u_to_j.spaces

  uh = zero(U)
  φh = zero(V_φ)
  j = Ref(0.0)
  u_to_j.cache.fwd_cache = (get_free_dof_values(uh), get_free_dof_values(φh), j)

  u_to_j.cache.plb_cache = (get_free_dof_values(zero(U)), get_free_dof_values(zero(V_φ)), ∂F∂u, ∂F∂φ)

  diff_order = get_diff_order(u_to_j) 
  u_to_j.cache.inc_obj_cache = build_inc_obj_cache(u_to_j.F,uh,φh,u_to_j.spaces,diff_order) 

  u_to_j.cache.cache_built = true
  u_to_j.cache.fwd_ran = false
  u_to_j.cache.bwd_ran = false

  u_to_j.cache
end
