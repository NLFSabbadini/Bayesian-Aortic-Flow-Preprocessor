include("TriMeshes.jl")
using .TriMeshes
using SparseArrays
using LinearAlgebra
using KrylovKit
using Base.Iterators
using HDF5
using Distributions


"""Convenience types"""
AbstractRn = AbstractVector{<:AbstractFloat}
AbstractRns = AbstractVector{<:AbstractRn}
AbstractRnn = AbstractMatrix{<:AbstractFloat}
AbstractRnns = AbstractVector{<:AbstractRnn}


"""Compute the least-squares optimal plane w^T x = 1 for points xs"""
function planarLeastSquares(xs::T)::Vector{Float64} where T <: AbstractRns
	return mapreduce(x -> x * transpose(x), +, xs) \ sum(xs)
end


"""Project points xs onto plane w^T x = 1 in place"""
function planarProjection!(w::Vector{Float64}, nodes::T)::Nothing where T <: AbstractRns
	p = w/dot(w,w)
	P = -p*transpose(w)
	nodes .+= Ref(p) .+ Ref(P) .* nodes
	return nothing
end


"""Model for the marginal covariance of surface nodes, given their radial normal"""
function marginalCovariance(n::T, sigma::Float64, tau::Float64) where T <: AbstractRn
	return Symmetric(tau^2*I + (sigma^2 - tau^2)*n*transpose(n)) #Get rid of numerical asymmetry
end


"""Compute the joint Gaussian distribution of N coefficients per basis vector,
obtained by probability scoring with entry-wise N-variate Gaussians of given means and covariances"""
function basisJointGaussianND(basis::T, mus::U, sigmas::V) where {T <: AbstractRnn, U <: AbstractRns, V <: AbstractRnns}
	d_x = length(mus[1])
	n_x = size(basis)[1]

	invSigmas_x = spzeros(d_x*n_x, d_x*n_x)
	blocks = [k:k+d_x-1 for k in 1:d_x:d_x*n_x]
	for (block, sigma) in zip(blocks, sigmas)
		invSigmas_x[block,block] .= inv(sigma)
	end

	B = kron(basis, Matrix(1.0I, d_x, d_x))
	C = transpose(sparse(cholesky(Symmetric(invSigmas_x)).L)) * B
	Sigma_phi = Symmetric(cholesky(Symmetric(transpose(C)*C)) \ I)
	mu_phi = *(Sigma_phi, transpose(B), invSigmas_x, reduce(vcat, mus))

	return MvNormal(mu_phi, Sigma_phi)
end


"""Given radial coordinates and harmonics for a surface, sample from the space of surfaces with a set minimum wavelength,
such that a Gaussian marginal distribution with set variance is approximated around each original vertex"""
function main(surfacePath::String, radialCoordsPath::String, harmonicsPath::String, outputPathTemp::String, 
edgeLength::Float64, minWavelength::Float64, radialUncertainty::Float64, numRealizations::Int64)::Nothing
	surface = TriMesh(surfacePath)
	normals, radii = h5open(radialCoordsPath, "r") do file
		(collect(eachcol(read(file["normals"]))), read(file["radii"]))
	end
	freqs, modes = h5open(harmonicsPath, "r") do file
		(read(file["freqs"]), read(file["modes"]))
	end

	maxFreq = (2*pi*edgeLength/minWavelength)^2
	basis = view(modes, :, filter!(i -> freqs[i] <= maxFreq, collect(1:length(freqs))))
	rho_phi = basisJointGaussianND(basis, 
		[fill(r, 1) for r in radii], #N-dimensional (position) vectors also supported!
		[fill(radialUncertainty^2, 1, 1) for r in radii]) #Varying uncertainty also supported!

	boundaryIds = boundaries(surface)
	boundaryPlanes = planarLeastSquares.(view.(Ref(surface.nodes), boundaryIds))

	for i in 0:numRealizations
		sampleWeights = i > 0 ? rand(rho_phi) : mean(rho_phi)
		sampleNodes = surface.nodes .+ normals .* (basis * sampleWeights .- radii)
		planarProjection!.(boundaryPlanes, view.(Ref(sampleNodes), boundaryIds)) #Just for safety
		save(TriMesh(sampleNodes, surface.triangles), replace(outputPathTemp, "%g"=>i))
	end
end


main(ARGS[1:4]..., parse.(Float64, ARGS[5:7])..., parse(Int64, ARGS[8]))
