include("TriMeshes.jl")
using .TriMeshes
using LinearAlgebra
using Distributions
using IterTools
using NPZ
using Printf


"""Convenience types"""
AbstractRn = AbstractVector{<:AbstractFloat}
AbstractRns = AbstractVector{<:AbstractRn}


"""Compute the cross Gram matrix for vector lists xs and ys, given kernel function k"""
function crossGramian(xs::T, ys::T, k::Function)::Matrix{Float64} where T <: AbstractRns
	K = Matrix{Float64}(undef, length(xs), length(ys))
	for i in 1:length(xs), j in 1:length(ys)
		K[i,j] = k(xs[i], ys[j])
	end
	return K
end


"""Cross Gram matrix for a distance kernel"""
function distanceCrossGramian(xs::T, ys::T)::Matrix{Float64} where T <: AbstractRns
	return crossGramian(xs, ys, (x, y) -> norm(x - y))
end


"""Cross Gram matrix for a Gaussian kernel, given some standard deviation"""
function gaussCrossGramian(xs::T, ys::T, sigma::Float64)::Matrix{Float64} where T <: AbstractRns
	return crossGramian(xs, ys, (x, y) -> exp(-0.5*norm(x - y)^2/sigma^2))
end


"""Ridge regularize a weakly positive definite matrix for numerical tractability,
using a grid search to approximate the minimal effective regularization parameter"""
function optimalWPDRidgeReg(M::Matrix{Float64}, n::Int64=1000)::Matrix{Float64}
	regs = [0; collect(logrange(2^(-52), maximum(eigen(M).values)*1e-7, n))]
	for reg in regs
		M_reg = M + reg*I
		if isposdef(M_reg)
			return M_reg
		end
	end
end


"""Ridge regularize a weakly positive definite matrix for numerical tractability,
using a grid search to approximate the effective regularization parameter r which minimizes
the reconstruction error ||X - M inv(M + r*I) X||, where X = [x1 x2 ... xn]"""
function optimalWPDRidgeReg(M::Matrix{Float64}, X::Matrix{Float64}, n::Int64=1000)::Matrix{Float64}
	regs = [0, collect(logrange(2^(-52), maximum(eigen(M).values)*1e-7, n))]
	errors = fill(Inf, length(regs))
	for (i, reg) in enumerate(regs)
		M_reg = M + reg*I
		if isposdef(M_reg)
			errors[i] = norm(X - *(M, inv(M_reg), X))
		end
	end
	return M + I*regs[argmin(errors)]
end


"""Model for the marginal covariance matrix of 4D Flow MRI vectors, as found in
O. Friman et. al. 2011 'Probabilistic 4D blood ﬂow tracking and uncertainty estimation'.
Dependence on v is kept as a place holder for future improved models"""
function marginalCovariance(v::T, sigma::Float64)::Matrix{Float64} where T <: AbstractRn
	return sigma^2 * [1 .5 .5; .5 1 .5; .5 .5 1]
end


"""Compute the joint covariance matrix for a set of vector valued random variables, given their marginal
covariance matrices and a scalar joint covariance matrix applying to the vector components"""
function jointVectorCovariance(vecCovs::Vector{Matrix{Float64}}, jointCompCov::Matrix{Float64}, separateComponents::Bool=true)::Matrix{Float64}
	d, n = (size(vecCovs[1])[1], size(jointCompCov)[1])
	S = Matrix{Float64}(undef, n*d, n*d)
	blocks = [(1:d).+d*k for k in 0:n-1]
	sqVecCovs = sqrt.(vecCovs)

	for i in 1:n, j in 1:n
		S[blocks[i], blocks[j]] = sqVecCovs[i] * sqVecCovs[j] * jointCompCov[i, j]
	end

	if separateComponents
		perm = vcat(range.(1:d, n*d, step=d)...)
		S = S[perm, perm]
	end

	return S
end


"""Construct a vector valued function where each entry is a different weighted sum of a set of Gaussians, 
given their radius sigma, their centers mu, and their weights for each entry"""
function sumOfGaussians(sigma::Float64, mu::T, W::Matrix{Float64})::Function where T <: AbstractRns
	function f(x::U)::Vector{Float64} where U <: AbstractRn
		fx = zeros(size(W)[2])
		for (mui, wi) in zip(mu, eachrow(W))
			fx += wi*exp(-0.5*norm(mui - x)^2/sigma^2)
		end
		return fx
	end
	return f
end


"""Write a .prof plain text file compatible with ANSYS, describing a static velocity B.C. in 3D"""
function writeAnsysProfile(path::String, xs::T, vs::T)::Nothing where T <: AbstractRns
	#Mind the spaces on the string below!
	content ="""
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
	)"""
	write(path, content)
	return nothing
end


"""Sample from an assumed joint distribution for 4D Flow MRI vectors and propagate to a mesh using the RBF method,
with no-flow B.C.s approximately enforced."""
function main(meshPath::String, vectorPath::String, outputPathTemp::String, sigma::Float64, numSamples::Int64)::Nothing
	vectorField = npzread(vectorPath)
	x = collect(eachcol(vectorField[1:3,:])) #positions
	n_x = length(x)
	r = sum(distanceCrossGramian(x, x))/n_x^2 #average distance

	mu_v = collect(eachcol(vectorField[4:6,:]))
	Sigma_v = optimalWPDRidgeReg(
		jointVectorCovariance(
			marginalCovariance.(mu_v, sigma),
			gaussCrossGramian(x, x, r)))
	rho_v = MvNormal(vcat(mu_v...), Sigma_v)

	mesh = TriMesh(meshPath)
	c = centroids(mesh)
	b = mesh.nodes[boundaries(mesh)[1]] #boundary nodes
	K = optimalWPDRidgeReg(gaussCrossGramian([x; b], [x; b], r))
	T = inv(K)[:, 1:n_x] #Transform from velocities to RBF weights, assuming no-flow b.c.s
	ws = T * reshape(hcat(mean(rho_v), rand(rho_v, numSamples)), n_x, :) #RBF weights for each component of each sample
	vs = gaussQuadrature(mesh, sumOfGaussians(r, [x; b], ws), Vector{Float64}) ./ areas(mesh) #mean velocity for each component of each sample on each cell

	for (i, rng) in enumerate([k:k+2 for k in 1:3:3*(numSamples+1)])
		writeAnsysProfile(replace(outputPathTemp, "%g"=>i-1), c, getindex.(vs, Ref(rng)))
	end
end


main(ARGS[1:3]..., parse(Float64, ARGS[4]), parse(Int64,ARGS[5]))
