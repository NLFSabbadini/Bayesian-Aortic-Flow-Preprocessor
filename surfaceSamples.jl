include("TriMeshes.jl")
using .TriMeshes
using SparseArrays
using LinearAlgebra
using KrylovKit
using Base.Iterators
using HDF5


"""Compute the least-squares optimal plane w^T x = 1 for points xs"""
function planarLeastSquares(xs::AbstractVector{<:AbstractVector{Float64}})::Vector{Float64}
	return mapreduce(x -> x * transpose(x), +, xs) \ sum(xs)
end


"""Project points xs onto plane w^T x = 1 in place"""
function planarProjection!(w::Vector{Float64}, nodes::AbstractVector{<:AbstractVector{Float64}})::Nothing
	p = w/dot(w,w)
	P = -p*transpose(w)
	nodes .+= Ref(p) .+ Ref(P) .* nodes
	return nothing
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
	basisIds = filter!(i -> freqs[i] <= maxFreq, collect(1:length(freqs)))
	basis = view(modes, :, basisIds)
	meanWeights = transpose(basis) * radii

	boundaryIds = boundaries(surface)
	boundaryPlanes = planarLeastSquares.(view.(Ref(surface.nodes), boundaryIds))

	for i in 0:numRealizations
		sampleWeights = i > 0 ? meanWeights + radialUncertainty * randn(length(basisIds)) : meanWeights
		sampleNodes = surface.nodes .+ normals .* (basis * sampleWeights .- radii)
		planarProjection!.(boundaryPlanes, view.(Ref(sampleNodes), boundaryIds))
		save(TriMesh(sampleNodes, surface.triangles), replace(outputPathTemp, "%g"=>i))
	end
end


main(ARGS[1:4]..., parse.(Float64, ARGS[5:7])..., parse(Int64, ARGS[8]))
