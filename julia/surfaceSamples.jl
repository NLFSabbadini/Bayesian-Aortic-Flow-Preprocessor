include("TriMeshes.jl")
using .TriMeshes
using SparseArrays
using LinearAlgebra
using KrylovKit
using Base.Iterators
using HDF5
using Distributions
using Kronecker
using KrylovKit


"""Convenience types"""
AbstractRn = AbstractVector{<:AbstractFloat}
AbstractRns = AbstractVector{<:AbstractRn}
AbstractRnn = AbstractMatrix{<:AbstractFloat}
AbstractRnns = AbstractVector{<:AbstractRnn}


"""Compute the least-squares optimal plane w^T x = 1 for points xs"""
function planarLeastSquares(xs::T)::Vector{Float64} where T <: AbstractRns
	return mapreduce(x -> x * transpose(x), +, xs) \ sum(xs)
end


"""Compute an affine basis for plane w^T x = 1"""
function planarBasis(w::Vector{Float64})::Tuple{Matrix{Float64}, Vector{Float64}}
	P = zeros(3, 2)
	P[:, 1] = normalize!(cross(w, [0, 0, 1]))
	P[:, 2] = cross(P[:,1], w/norm(w))
	return P, w/dot(w,w)
end


"""Compute the bayesian posterior distribution for a given affine basis Aw+b conditioned on x, 
assuming a diagonal, zero mean Gaussian and a diagonal Gaussian measurement likelihood"""
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
	Sigma = U*transpose(U)
	mu = *(Sigma, transpose(A), (x-b)./sigmasM.^2)
	return MvNormal(mu, Sigma)
end


"""Sample from the space of surfaces with a set minimum wavelength, such that a Gaussian marginal distribution with 
set variance is approximated around each original vertex"""
function main(surfacePath::String, harmonicsPath::String, outputPathTemp::String, 
edgeLength::Float64, surfaceUncertainty::Float64, numRealizations::Int64)::Nothing
	surface = TriMesh(surfacePath)
	n = 3*length(surface.nodes)

	#Interior harmonic basis
	basis, frequencies = h5open(harmonicsPath, "r") do file
		(kron(read(file["harmonics"]),Matrix(1.0I,3,3)), 
			kron(read(file["frequencies"]),ones(3)))
	end
	offset = zeros(n)

	#Boundary harmonic basis (disable if using unconstrained harmonics)
	boundaryFrequencies, extendedBoundaryHarmonics = h5open(harmonicsPath, "r") do file
		ids = [start:stop for (start, stop) in read(file["boundaryHarmonicIds"])]
		(kron(read(file["boundaryFrequencies"]),ones(2)),
			view.(Ref(read(file["extendedBoundaryHarmonics"])),:,ids))
	end
	boundaryBases = []
	for (H,ids) in zip(extendedBoundaryHarmonics,boundaries(surface))
		P, p = planarBasis(planarLeastSquares(surface.nodes[ids]))
		offset += kron(H[:,1],sqrt(length(ids))*p) #extension of constant boundary modes, with boundaries set at respective plane offsets 
		push!(boundaryBases, kron(H,P))
	end
	basis = hcat(basis, boundaryBases...)
	frequencies = vcat(frequencies, boundaryFrequencies)
	
	#Distribution
	alpha, beta = (0.028677245760023833, 6.212056127641208) #prior power law parameters
	rho = posteriorDistribution(basis, offset, sqrt(alpha)*frequencies.^(-beta/2), fill(surfaceUncertainty, n), reduce(vcat, surface.nodes))
	
	#Sampling
	for i in 0:numRealizations
		sampleWeights = i > 0 ? rand(rho) : mean(rho)
		sampleNodes = collect(partition(basis*sampleWeights+offset,3))
		save(TriMesh(sampleNodes, surface.triangles), replace(outputPathTemp, "%g"=>i))
	end
end


main(ARGS[1:3]..., parse.(Float64, ARGS[4:5])..., parse(Int64, ARGS[6]))
