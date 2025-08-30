include("TriMeshes.jl")
using .TriMeshes
using LinearAlgebra
using Base.Iterators
using HDF5


"""Sample weights from a given multivariate Gaussian distribution and construct triangular meshes using a given affine basis"""
function main(basisPath::String, distributionPath::String, outputPathTemp::String, numSamples::Int64)::Nothing
	basis, offset, triangles, lsqWeights = h5open(basisPath, "r") do file
		(read(file["basis"]), read(file["offset"]), collect(eachcol(read(file["triangles"]))), read(file["leastSquares"]))
	end

	mu, U = h5open(distributionPath, "r") do file
		(read(file["mean"]), read(file["covarianceFactor"]))
	end
	
	#Sampling
	for i in 0:numSamples
		sampleWeights = i > 0 ? U * randn(size(U)[1]) + mu : lsqWeights
		sampleNodes = collect(partition(basis*sampleWeights+offset, 3))
		save(TriMesh(sampleNodes, triangles), replace(outputPathTemp, "%"=>i))
	end
end


main(ARGS[1:3]..., parse.(Int64, ARGS[4]))
