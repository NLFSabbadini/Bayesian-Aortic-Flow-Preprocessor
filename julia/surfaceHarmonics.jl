include("TriMeshes.jl")
using .TriMeshes
using LinearAlgebra
using KrylovKit
using HDF5


"""Compute the unconstrained graph harmonics for a given triangle mesh, 
using the Lanczos algorithm with given Krylov subspace dimension"""
function unconstrainedGraphHarmonics(mesh::TriMesh, krylovDim::Int64)::Tuple{Vector{Float64}, Vector{Vector{Float64}}}
	L = laplacian(mesh, :graph)
	freqs, modes, _ = eigsolve(L, length(mesh.nodes), krylovDim, :SR, krylovdim=krylovDim)
	return freqs, modes
end


"""Compute the Dirichlet graph harmonics for a given triangle mesh,
using the Lanczos algorithm with given Krylov subspace dimension"""
function dirichletGraphHarmonics(mesh::TriMesh, krylovDim::Int64)::Tuple{Vector{Float64}, Vector{Vector{Float64}}}
	L = laplacian(mesh, :graph, dirichletNodes=vcat(boundaries(mesh)), homogeneous=true)
	freqs, modes, _ = eigsolve(L, length(mesh.nodes), krylovDim, :SR, krylovdim=krylovDim)
	return freqs, modes
end


"""Approximate a number of harmonic functions and frequencies of a triangular mesh, using the Lanczos algorithm"""
function main(surfacePath::String, modePath::String, krylovDim::Int64)::Nothing
	freqs, modes = unconstrainedGraphHarmonics(TriMesh(surfacePath), krylovDim)
	h5open(modePath, "w") do file #save eigenvalues and eigenvectors
		write(file, "freqs", freqs, "modes", stack(modes))
	end
end


main(ARGS[1:2]..., parse(Int64,ARGS[3]))
