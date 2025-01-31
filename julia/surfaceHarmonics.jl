include("TriMeshes.jl")
using .TriMeshes
using KrylovKit
using HDF5


"""Approximate a number of harmonic functions and frequencies of a triangular mesh, using the Lanczos algorithm"""
function main(triMeshPath::String, modePath::String, krylovDim::Int64)::Nothing
	L = laplacian(TriMesh(triMeshPath)) #load surface mesh and compute graph Laplacian
	freqs, modes, info = eigsolve(L, size(L)[1], krylovDim, :SR, krylovdim=krylovDim) #Lanczos algorithm with Krylov subspace dimension krylovDim
	h5open(modePath, "w") do file #save eigenvalues and eigenvectors
		write(file, "freqs", freqs, "modes", stack(modes))
	end
end


main(ARGS[1:2]..., parse(Int64,ARGS[3]))
