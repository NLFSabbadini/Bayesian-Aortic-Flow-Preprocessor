include("TriMeshes.jl")
using .TriMeshes
using LinearAlgebra
using SparseArrays
using KrylovKit
using HDF5


"""Compute the least-squares optimal plane w^T x = 1 for points xs"""
function planarLeastSquares(xs::T)::Vector{Float64} where T <: AbstractVector{<:AbstractVector{<:AbstractFloat}}
	return mapreduce(x -> x * transpose(x), +, xs) \ sum(xs)
end


"""Compute an affine basis for plane w^T x = 1"""
function planarBasis(w::Vector{Float64})::Tuple{Matrix{Float64}, Vector{Float64}}
	P = zeros(3, 2)
	P[:, 1] = normalize!(cross(w, [0, 0, 1]))
	P[:, 2] = cross(P[:,1], w/norm(w))
	return P, w/dot(w,w)
end


"""Compute a number of low frequency graph harmonics and corresponding frequencies 
for a given trianglar mesh, using the Lanczos algorithm"""
function unconstrainedGraphHarmonics(mesh::TriMesh, harmonicsRange::Float64, krylovMargin::Int64)::Tuple{Vector{Float64}, Vector{Float64}, Matrix{Float64}}
	numNodes = length(mesh.nodes)
	numHarmonics = Int64(ceil(numNodes*harmonicsRange))
	L = laplacian(mesh, :graph)
	eigenvalues, harmonics, _ = eigsolve(L, numNodes, numHarmonics, :LR, krylovdim=numHarmonics+krylovMargin)
	basis = kron(stack(harmonics), Matrix(1.0I, 3, 3))
	offset = zeros(3*numNodes)
	frequencies = kron(sqrt.(-eigenvalues), ones(3))
	return basis, offset, frequencies
end


"""Compute the graph harmonics and corresponding frequencies for a loop of length n"""
function loopGraphHarmonics(n::Int64)::Tuple{Vector{Float64}, Matrix{Float64}}
	kmaxcos = div(n,2)
	kmaxsin = div(n,2) - (iseven(n) ? 1 : 0)

	Vcos = zeros(n, kmaxcos + 1)
	Vsin = zeros(n, kmaxsin)

	for k in 0:kmaxcos
		Vcos[:, k+1] .= normalize!(cos.(2*pi/n*k*(1:n))) 
	end
	for k in 1:kmaxsin
		Vsin[:, k] .= normalize!(sin.(2*pi/n*k*(1:n)))
	end

	frequencies = 2*pi/n*vcat(0:kmaxcos, 1:kmaxsin)

	return frequencies, hcat(Vcos, Vsin)
end


"""Compute a number of low frequency Dirichlet graph harmonics and corresponding frequencies
for a given triangle mesh, including harmonic extensions into the interior of the boundary harmonics,
using the Lanczos algorithm"""
function dirichletGraphHarmonics(mesh::TriMesh, harmonicsRange::Float64, krylovMargin::Int64)
	numNodes = length(mesh.nodes)
	boundaryIds = boundaries(mesh)
	boundaryIdsCat = vcat(boundaryIds...)
	interiorIds = setdiff(1:numNodes, boundaryIdsCat)
	numInteriorNodes = length(interiorIds)

	LHD = laplacian(mesh, :graph, dirichletNodes=boundaryIdsCat, homogeneous=true, removeBoundaries=true) #homogeneous Dirichlet Laplacian
	numHarmonics = Int64(ceil(numInteriorNodes*harmonicsRange))
	eigenvalues, interiorHarmonics, _ = eigsolve(LHD, numInteriorNodes, numHarmonics, :LR, krylovdim=numHarmonics+krylovMargin)
	interiorFrequencies = kron(sqrt.(-eigenvalues), ones(3))
	maxInteriorFrequency = maximum(interiorFrequencies)
	interiorBasis1D = zeros(numNodes, length(interiorHarmonics))
	interiorBasis1D[interiorIds, :] .= stack(interiorHarmonics) 
	interiorBasis = kron(interiorBasis1D, Matrix(1.0I, 3, 3))

	LD = laplacian(mesh, :graph, dirichletNodes=boundaryIdsCat) #non-homogeneous Dirichlet Laplacian
	In = spdiagm(ones(numNodes))
	boundaryBases = []
	boundaryOffsets = []
	boundaryFrequencies = []
	for ids in boundaryIds
		f, h = loopGraphHarmonics(length(ids))
		js = (1:size(h)[2])[maxInteriorFrequency .> f]
		P, p = planarBasis(planarLeastSquares(mesh.nodes[ids]))
		H = LD \ (In[:,ids] * h[:,js])
		push!(boundaryBases, kron(H, P))
		push!(boundaryOffsets, kron(H[:,1], sqrt(length(ids))*p))
		push!(boundaryFrequencies, kron(f[js], ones(2)))
	end

	basis = hcat(interiorBasis, boundaryBases...)
	offset = sum(boundaryOffsets)
	frequencies = vcat(interiorFrequencies, boundaryFrequencies...)

	return basis, offset, frequencies
end


"""Compute a range of low frequency (Dirichlet) graph harmonics and corresponding frequencies/approximate wavelengths
for a given triangle mesh, including harmonic extensions into the interior of the boundary harmonics,
using the Lanczos algorithm"""
function main(meshPath::String, outputPath::String)::Nothing
	harmonicsRange = 0.1 #bottom x fraction (where vast majority of signal energy lies)
	krylovMargin = 30 #number of buffer dimensions for the Krylov subspace methods, to guarantee accuracy

	mesh = TriMesh(meshPath)
	basis, offset, frequencies = dirichletGraphHarmonics(mesh, harmonicsRange, krylovMargin)
	lsqWeigths = (transpose(basis) * basis) \ (transpose(basis) * (reduce(vcat, mesh.nodes) - offset))

	h5open(outputPath, "w") do file #save eigenvalues and eigenvectors
		write(file, "basis", basis, "offset", offset, "frequencies", frequencies, "triangles", stack(mesh.triangles), "leastSquares", lsqWeigths)
	end
end


main(ARGS[1:2]...)
