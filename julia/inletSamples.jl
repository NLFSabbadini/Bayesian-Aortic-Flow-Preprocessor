include("TriMeshes.jl")
using .TriMeshes
using LinearAlgebra
using Distributions
using IterTools
using NPZ
using Printf
using ZipFile


"""Convenience types"""
AbstractRn = AbstractVector{<:AbstractFloat}
AbstractRns = AbstractVector{<:AbstractRn}
AbstractRnn = AbstractMatrix{<:AbstractFloat}
AbstractRnns = AbstractVector{<:AbstractRnn}


"""Compute the cross Gram matrix for vector lists xs and ys, given kernel function k"""
function crossGramian(xs::T, ys::U, k::Function)::Matrix{Float64} where {T<:AbstractRns, U<:AbstractRns}
	K = Matrix{Float64}(undef, length(xs), length(ys))
	for i in 1:length(xs), j in 1:length(ys)
		K[i,j] = k(xs[i], ys[j])
	end
	return K
end


"""Cross Gram matrix for distance kernel"""
function distanceCrossGramian(xs::T, ys::U)::Matrix{Float64} where {T<:AbstractRns, U<:AbstractRns}
	return crossGramian(xs, ys, (x, y) -> norm(x - y))
end


"""Cross Gram matrix for Gaussian kernel with std sigma"""
function gaussCrossGramian(xs::T, ys::U, sigma::Float64)::Matrix{Float64} where {T<:AbstractRns, U<:AbstractRns}
	return crossGramian(xs, ys, (x, y) -> exp(-0.5*norm(x - y)^2/sigma^2))
end


"""Ridge regularize a weakly positive definite matrix for numerical tractability,
using a grid search to approximate the minimal effective regularization parameter"""
function optimalWPDRidgeReg(M::Matrix{Float64}, n::Int64=1000)::Matrix{Float64}
	regs = [0; logrange(2^(-52), maximum(eigen(M).values)*1e-7, n)] #[0, machine epsilon ... dynamic range 1e7]
	for reg in regs
		M_reg = M + reg*I
		if isposdef(M_reg)
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


"""Construct the RBF interpolant function for vectors ys = [y1 y2 ... yn]^T at positions xs"""
function RBFInterpolant(xs::T, ys::U, r::Float64)::Function where {T<:AbstractRns, U<:AbstractRnn}
	K = optimalWPDRidgeReg(gaussCrossGramian(xs, xs, r)) #Ridge regularization in case of ill-conditioning
	W = K \ ys

	function interpolant(x::V)::Vector{Float64} where V<:AbstractRn
		y = zeros(size(W)[2])
		for (xi, wi) in zip(xs, eachrow(W))
			y += wi*exp(-0.5*norm(xi - x)^2/r^2)
		end
		return y
	end

	return interpolant
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


"""Sample from an assumed joint distribution for 4D Flow MRI vectors and interpolate to a mesh using the RBF method"""
function main(meshPath::String, vectorPath::String, outputPath::String, sigma::Float64, numSamples::Int64)
	mesh = TriMesh(meshPath)
	vectorField = npzread(vectorPath)
	xs = collect(eachcol(vectorField[1:3,:])) #positions

	#Construct velocity distribution at datapoints
	lc = 1. #correlation length, insert heuristic
	mu_v = reshape(vectorField[4:6,:],:)
	Sigma_v = optimalWPDRidgeReg( #ridge regularization in case of ill-conditioning
		jointVectorCovariance(
			marginalCovariance.(collect.(partition(mu_v, 3)), sigma),
			gaussCrossGramian(xs, xs, lc)))
	rho_v = MvNormal(mu_v, Sigma_v)
	
	#Sample distribution and rearrange as [x1 y1 z1 ... xn yn zn]
	vs = zeros(length(xs), 3*(numSamples+1)) #length(xs) + length(vcat(boundaries(mesh)...)) to include no-slip forcing
	vs[:,1:3] .= transpose(reshape(mu_v, 3, :))
	for i in 1:numSamples
		vs[:, 3*i.+(1:3)] .= transpose(reshape(rand(rho_v), 3, :))
	end

	#Construct RBF interpolant and numerically average over cells
	r = unique(sort(reshape(distanceCrossGramian(xs, xs), :)))[2]/2 #this works, but find better heuristic?
	vc = 1 ./ areas(mesh) .* gaussQuadrature( #RBFInterpolant(vcat(xs, boundaries(mesh)...), vs, r) to include no-slip forcing
		mesh, RBFInterpolant(xs, vs, r), Vector{Float64})

	#Write Ansys .prof files
	cs = centroids(mesh)
	zip = ZipFile.Writer(outputPath)
	for (i, rng) in enumerate([k:k+2 for k in 1:3:3*(numSamples+1)])
		profile = ZipFile.addfile(zip, "$(i).prof")
		write(profile, ansysProfile(cs, getindex.(vc, Ref(rng))))
	end
	close(zip)
end


main(ARGS[1:3]..., parse(Float64, ARGS[4]), parse(Int64,ARGS[5]))
