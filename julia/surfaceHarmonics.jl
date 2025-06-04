include("TriMeshes.jl")
using .TriMeshes
using LinearAlgebra
using SparseArrays
using KrylovKit
using HDF5


"""Compute a numer of low frequency graph harmonics and corresponding frequencies/approximate wavelengths
for a given triangle mesh, using the Lanczos algorithm"""
function unconstrainedGraphHarmonics(mesh::TriMesh, harmonicsRange::Float64, krylovMargin::Int64)::Tuple{Vector{Float64}, Vector{Float64}, Matrix{Float64}}
	numNodes = length(mesh.nodes)
	numHarmonics = Int64(ceil(numNodes*harmonicsRange))
	L = laplacian(mesh, :graph)
	eigenvalues, harmonics, _ = eigsolve(L, numNodes, numHarmonics, :LR, krylovdim=numHarmonics+krylovMargin)
	harmonics = stack(harmonics)
	frequencies = sqrt.(-eigenvalues)
	wavelengths = 2*pi./frequencies
	return frequencies, wavelengths, harmonics
end


"""Compute the graph harmonics for a loop of length n"""
function loopGraphHarmonics(n::Int64)::Tuple{Vector{Float64}, Vector{Float64}, Matrix{Float64}}
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
	wavelengths = 2*pi./frequencies

	return frequencies, wavelengths, hcat(Vcos, Vsin)
end


"""Compute a number of low frequency Dirichlet graph harmonics and corresponding frequencies/approximate wavelengths
for a given triangle mesh, including harmonic extensions into the interior of the boundary harmonics,
using the Lanczos algorithm with given Krylov subspace dimension"""
function dirichletGraphHarmonics(mesh::TriMesh, harmonicsRange::Float64, krylovMargin::Int64)
	numNodes = length(mesh.nodes)
	boundaryIds = boundaries(mesh)
	boundaryIdsCat = vcat(boundaryIds...)
	numInteriorNodes = numNodes - length(boundaryIdsCat)

	LHD = laplacian(mesh, :graph, dirichletNodes=boundaryIdsCat, homogeneous=true, removeBoundaries=true) #homogeneous Dirichlet Laplacian
	numHarmonics = Int64(ceil(numInteriorNodes*harmonicsRange))
	eigenvalues, harmonicsInterior, _ = eigsolve(LHD, numInteriorNodes, numHarmonics, :LR, krylovdim=numHarmonics+krylovMargin)
	frequencies = sqrt.(-eigenvalues)
	wavelengths = 2*pi./frequencies
	keepRange = setdiff(1:numNodes, boundaryIdsCat) #include boundary nodes
	harmonics = zeros(numNodes, length(harmonicsInterior))
	harmonics[keepRange,:] .= stack(harmonicsInterior)

	LD = laplacian(mesh, :graph, dirichletNodes=boundaryIdsCat) #non-homogeneous Dirichlet Laplacian
	In = spdiagm(ones(numNodes))
	boundaryFrequencies = []
	boundaryWavelengths = []
	extendedBoundaryHarmonics = []
	numBoundaryHarmonics = []
	for ids in boundaryIds
		f, w, h = loopGraphHarmonics(length(ids))
		js = (1:size(h)[2])[w .> minimum(wavelengths)]
		push!(numBoundaryHarmonics, length(js))
		push!(boundaryFrequencies, f[js])
		push!(boundaryWavelengths, w[js])
		push!(extendedBoundaryHarmonics, LD \ (In[:,ids] * h[:,js]))
	end
	boundaryFrequencies = vcat(boundaryFrequencies...)
	boundaryWavelengths = vcat(boundaryWavelengths...)
	extendedBoundaryHarmonics = hcat(extendedBoundaryHarmonics...)
	boundaryHarmonicIds = range.(accumulate(+,[1; numBoundaryHarmonics[1:end-1]]), accumulate(+, numBoundaryHarmonics))

	return (frequencies, wavelengths, harmonics, boundaryFrequencies, 
		boundaryWavelengths, extendedBoundaryHarmonics, boundaryHarmonicIds)
end


"""Compute a range of low frequency (Dirichlet) graph harmonics and corresponding frequencies/approximate wavelengths
for a given triangle mesh, including harmonic extensions into the interior of the boundary harmonics,
using the Lanczos algorithm"""
function main(meshPath::String, harmonicsPath::String)::Nothing
	harmonicsRange = 0.1 #bottom x percent (where vast majority of signal energy lies)
	krylovMargin = 30 #number of buffer dimensions for the Krylov subspace methods, to guarantee accuracy

	mesh = TriMesh(meshPath)
	out = dirichletGraphHarmonics(mesh, harmonicsRange, krylovMargin)

	h5open(harmonicsPath, "w") do file #save eigenvalues and eigenvectors
		write(file, "frequencies", out[1], "wavelengths", out[2], "harmonics", out[3])
		write(file, "boundaryFrequencies", out[4], "boundaryWavelengths", out[5], #disable if using unconstrained harmonics:
			"extendedBoundaryHarmonics", out[6], "boundaryHarmonicIds", out[7])
	end
end


main(ARGS[1:2]...)
