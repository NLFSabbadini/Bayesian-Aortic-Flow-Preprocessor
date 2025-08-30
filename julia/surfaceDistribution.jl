include("TriMeshes.jl")
using .TriMeshes
using LinearAlgebra
using HDF5


"""Convenience types"""
AbstractRn = AbstractVector{<:AbstractFloat}
AbstractRnn = AbstractMatrix{<:AbstractFloat}


"""Compute the bayesian posterior distribution for a given affine basis Aw+b conditioned on x, 
assuming a diagonal, zero mean Gaussian prior and a diagonal Gaussian measurement likelihood"""
function posteriorDistribution(A::T1, b::T2, sigmasP::T3, sigmasM::T4, x::T5) where {T1 <: AbstractRnn, T2 <: AbstractRn, T3 <: AbstractRn, T4 <: AbstractRn, T5 <: AbstractRn}
	n, m = size(A)
	Z = zeros(n+m, m)
	Z[1:n,:] .= A
	for i in 1:n
		Z[i,:] ./= sigmasM[i]
	end
	for i in 1:m
		Z[n+i,i] = 1/sigmasP[i]
	end
	U = qr(Z).R \ I
	mu = *(U, transpose(U), transpose(A), (x-b)./sigmasM.^2)
	return mu, U
end


"""Construct Bayesian posterior for a given triangular surface mesh and corresponding Dirichlet harmonic basis"""
function main(surfacePath::String, basisPath::String, outputPath::String, surfaceUncertainty::Float64)::Nothing
	surface = TriMesh(surfacePath)
	basis, offset, frequencies = h5open(basisPath, "r") do file
		(read(file["basis"]), read(file["offset"]), read(file["frequencies"]))
	end
	
	#Distribution
	alpha, beta = (0.028677245760023833, 6.212056127641208) #prior power law parameters
	mu, U = posteriorDistribution(basis, offset, sqrt(alpha)*frequencies.^(-beta/2), fill(surfaceUncertainty, size(basis)[1]), reduce(vcat, surface.nodes))
	h5open(outputPath, "w") do file #save distribution
		write(file, "mean", mu, "covarianceFactor", U)
	end
end


main(ARGS[1:3]..., parse.(Float64, ARGS[4]))
