include("TriMeshes.jl")
using .TriMeshes
using LinearAlgebra
using KrylovKit
using Distributions
using IterTools
using NPZ
using Printf
using ZipFile
using SparseArrays
using CairoMakie
using IterTools


"""Convenience types"""
AbstractRn = AbstractVector{<:AbstractFloat}
AbstractRns = AbstractVector{<:AbstractRn}
AbstractRnn = AbstractMatrix{<:AbstractFloat}
AbstractRnns = AbstractVector{<:AbstractRnn}


"""Compute the cross Gram matrix for vector lists xs and ys, given kernel function k"""
function gramian(xs::T, k::Function)::Matrix{Float64} where T<:AbstractRns
	K = Matrix{Float64}(undef, length(xs), length(xs))
	for i in 1:length(xs)
		K[i,i] = k(xs[i],xs[i])
	end
	for i in 2:length(xs), j in 1:i-1
		K[i,j] = k(xs[i], xs[j])
		K[j,i] = k(xs[i], xs[j])
	end
	return K
end


"""Compute the cross Gram matrix for vector lists xs and ys, given kernel function k"""
function crossGramian(xs::T, ys::U, k::Function)::Matrix{Float64} where {T<:AbstractRns, U<:AbstractRns}
	K = Matrix{Float64}(undef, length(xs), length(ys))
	for i in 1:length(xs), j in 1:length(ys)
		K[i,j] = k(xs[i], ys[j])
	end
	return K
end


"""Ridge regularize a weakly positive definite matrix for numerical tractability,
using a grid search to approximate the minimal effective regularization parameter"""
function optimalWPDRidgeReg(M::Matrix{Float64}, n::Int64=1000)::Matrix{Float64}
	regs = [0; logrange(2^(-52), maximum(eigen(M).values)*1e-7, n)] #[0, machine epsilon ... dynamic range 1e7]
	for reg in regs
		M_reg = M + reg*I
		if isposdef(M_reg)
			println(reg)
			return M_reg
		end
	end
end


"""Model for the marginal covariance matrix of 4D Flow MRI vectors, as found in
O. Friman et. al. 2011 'Probabilistic 4D blood ﬂow tracking and uncertainty estimation'.
Dependence on v is kept as a place holder for future improved models"""
function marginalCovariance(v::T, sigma::Float64)::Matrix{Float64} where T<:AbstractRn
	return sigma^2 * [1 .5 .5; .5 1 .5; .5 .5 1]
end


"""Compute the joint covariance matrix for a set of vector valued random variables, given their marginal
covariance matrices and a joint covariance matrix for the scalar components"""
function jointVectorCovariance(vecCovs::Vector{Matrix{Float64}}, jointCompCov::Matrix{Float64})::Matrix{Float64}
	d, n = (size(vecCovs[1])[1], size(jointCompCov)[1])
	S = Matrix{Float64}(undef, n*d, n*d)
	blocks = [(1:d).+d*k for k in 0:n-1]
	sqVecCovs = sqrt.(vecCovs)

	for i in 1:n, j in 1:n
		S[blocks[i], blocks[j]] = sqVecCovs[i] * sqVecCovs[j] * jointCompCov[i, j]
	end

	return S
end


"""Generate .prof file contents compatible with ANSYS, describing a static velocity B.C. in 3D"""
function ansysProfile(xs::T, vs::U)::String where {T<:AbstractRns, U<:AbstractRns}
	return """
	((inlet point $(length(xs)))
	    (x
	        $(join([@sprintf("%.12E", x[1]) for x in xs], " "))
	    )
	    (y
	        $(join([@sprintf("%.12E", x[2]) for x in xs], " "))
	    )
	    (z
	        $(join([@sprintf("%.12E", x[3]) for x in xs], " "))
	    )
	    (v-x
	        $(join([@sprintf("%.12E", v[1]) for v in xs], " "))
	    )
	    (v-y
	        $(join([@sprintf("%.12E", v[2]) for v in xs], " "))
	    )
	    (v-z
	        $(join([@sprintf("%.12E", v[3]) for v in xs], " "))
	    )
	)""" #Spaces in string required for indent!
end


function planarLeastSquares(xs::T)::Vector{Float64} where T <: AbstractRns
	return mapreduce(x -> x * transpose(x), +, xs) \ sum(xs)
end


function planarProjection(w::Vector{Float64}, xs::T)::Vector{Vector{Float64}} where T <: AbstractRns
	p = w/dot(w,w)
	P = I - w*transpose(p)
	return Ref(P) .* xs .+ Ref(p)
end


"""Compute an affine basis for plane w^T x = 1"""
function planarBasis(w::Vector{Float64})::Matrix{Float64}
	P = zeros(3, 2)
	P[:, 1] = normalize!(cross(w, [0, 0, 1]))
	P[:, 2] = cross(P[:,1], w/norm(w))
	return transpose(P)
end


"""Sample from an assumed joint distribution for 4D Flow MRI vectors and interpolate to a mesh using the RBF method"""
function main(meshPath::String, vectorPath::String, outputPath::String, sigma::Float64, numSamples::Int64)
	mesh = TriMesh(meshPath)
	vectorField = npzread(vectorPath)
	n = size(vectorField)[2]

	#Construct posterior distribution
	mu_l = reshape(vectorField[4:6,:],:) #likelihood mean
	Sigma_l = zeros(3*n,3*n) #likelihood covariance
	for block in collect.(partition(1:3*n, 3))
		Sigma_l[block,block] .= marginalCovariance(ones(3), sigma^2)
	end
	rho = MvNormal(mu_l, Symmetric(Sigma_l)) #uniform prior for now

	#Set up masked RBF interpolation
	xs = collect(eachcol(vectorField[1:3,:]))
	ps, W = gaussQuadrature(mesh) #quadrature points and weight matrix
	w = planarLeastSquares(xs) #inlet plane

	maskMesh = TriMesh(planarProjection(w, mesh.nodes[boundaries(mesh)[1]])) #mesh from projected boundary loop
	maskBnd = boundaries(maskMesh)[1]
	maskInt = setdiff(1:length(maskMesh.nodes), maskBnd)
	L = laplacian(maskMesh, :mesh, dirichletNodes=maskBnd, homogeneous=true, removeBoundaries=true)
	_, vecs, _ = eigsolve(L, length(maskInt), 1, :LR, krylovdim=31) #solve for fundamental harmonic h0
	h0 = zeros(length(mesh.nodes))
	h0[maskInt] .= abs.(vecs[1])
	mask = ys -> linearInterpolation(maskMesh, h0, planarProjection(w, ys))
	maskAtXs = mask(xs)
	keep = maskAtXs .> 0.1*maximum(h0) #heuristic to prevent ill conditioning
	maskAtPs = mask(ps)

	r = unique(sort(reshape(gramian(xs[keep], (x,y) -> norm(x-y)), :)))[2] #this works, but find better heuristic?
	rbfKernel = (x,y) -> exp(-0.5*norm(x - y)^2/r^2)
	K = (maskAtXs[keep]*transpose(maskAtXs[keep])) .* gramian(xs[keep], rbfKernel)
	T = (maskAtPs*transpose(maskAtXs[keep])) .* crossGramian(ps, xs[keep], rbfKernel)
	Linv = cholesky(K).L \ I #inv(K) = inv(L L^T) = inv(L)^T inv(L)

	#Sample distribution and interpolate
	vSamples = zeros(n, 3*(numSamples+1))
	vSamples[:,1:3] .= transpose(reshape(mean(rho), 3, :))
	for i in 1:numSamples
		vSamples[:, 3*i.+(1:3)] .= transpose(reshape(rand(rho), 3, :))
	end
	iSamples = *(W, T, transpose(Linv), Linv, vSamples[keep,:]) ./ areas(mesh)

	#Write Ansys .prof files
	cs = centroids(mesh)
	zip = ZipFile.Writer(outputPath)
	for (i, rng) in enumerate([k:k+2 for k in 1:3:3*(numSamples+1)])
		profile = ZipFile.addfile(zip, "$(i).prof")
		write(profile, ansysProfile(cs, eachrow(iSamples[:,rng])))
	end
	close(zip)
end


main(ARGS[1:3]..., parse(Float64, ARGS[4]), parse(Int64,ARGS[5]))
