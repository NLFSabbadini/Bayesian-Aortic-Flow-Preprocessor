include("TriMeshes.jl")
using .TriMeshes
using KrylovKit
using LinearAlgebra
using SparseArrays
using IterTools
using HDF5


"""Compute the least-squares optimal plane w^T x = 1 for points xs"""
function planarLeastSquares(xs::Vector{Vector{Float64}})::Vector{Float64} 
	return mapreduce(x -> x * transpose(x), +, xs) \ sum(xs)
end


"""Project points xs onto plane w^T x = 1"""
function planarProjection(w::Vector{Float64}, nodes::Vector{Vector{Float64}})::Vector{Vector{Float64}}
	p = w/dot(w,w)
	P = I - p*transpose(w)
	return Ref(p) .+ Ref(P) .* nodes
end


"""Compute Cartesian coordinates of 3D points xs on plane w^T x = 1, such that x1 lies on the first axis"""
function planarCoords(w::Vector{Float64}, nodes)::Vector{Vector{Float64}}
	P = zeros(2, 3)
	P[1, :] = normalize!(cross(w, [0, 0, 1]))
	P[2, :] = cross(P[1, :], w/norm(w))
	return Ref(P) .* nodes
end


"""Determine if 2D chains (a, b) and (c, d) intersect"""
function checkSegmentIntersection2D(a::Vector{Float64}, b::Vector{Float64}, c::Vector{Float64}, d::Vector{Float64})::Bool
	cross2D(a, b) = a[1]*b[2] - a[2]*b[1] #2D cross product
	orient(a, b, c) = cross2D(a, b) + cross2D(b, c) + cross2D(c, a)
	return orient(a, b, c) * orient(a, b, d) <= 0 && orient(c, d, a) * orient(c, d, b) <= 0
end


"""Compute the segment indices for the first intersection of 2D directed polygonal chain1 with chain2"""
function firstIntersection2D(chain1::Vector{Vector{Float64}}, chain2::Vector{Vector{Float64}})::Tuple{Int64,Int64}
	for (i, (a, b)) in enumerate(partition(chain1, 2, 1)), (j, (c, d)) in enumerate(partition(chain2, 2, 1))
		if checkSegmentIntersection2D(a, b, c, d)
			return i, j
		end
	end
	return length(chain1), length(chain2) #default value for no intersection
end


"""Given a cyclic list of 2D directed polygonal chains that each intersect with the next at least once, 
extract the node indices for the 'joint polygon' obtained by switching to the next chain at the first encuntered intersection"""
function jointPolyIndices2D(chains::Vector{Vector{Vector{Float64}}})::Vector{UnitRange{Int64}}
	ranges = range.(1, length.(chains))
	for (i, j) in partition([1:length(chains); 1], 2, 1)
		endRangei, endReverseRangej = firstIntersection2D(chains[i], reverse(chains[j]))
		beginRangej = length(chains[j]) - endReverseRangej + 1
		ranges[i] = ranges[i][begin]:endRangei
		ranges[j] = beginRangej:ranges[j][end]
	end
	return ranges
end


"""Iteratively apply jointPolyIndices2D to planar projections of a list of 3D directed polygonal chains, 
such that the extracted 'joint polygon' doesn't self intersect when projected to its least-squares optimal plane. 
Also return the 'loose ends' not included in the joint polygon, grouped by intersection."""
function robustJointPoly3D(chains::Vector{Vector{Vector{Float64}}})::Tuple{Vector{Vector{Float64}}, Vector{Vector{Vector{Vector{Float64}}}}}
	ranges = range.(1, length.(chains))
	while true
		w = planarLeastSquares(vcat(getindex.(chains, ranges)...)) #optimal plane for inner boundary
		chainsProj = planarCoords.(Ref(w), chains)
		newRanges = jointPolyIndices2D(chainsProj)
		if newRanges == ranges
			break
		end
		ranges = newRanges
	end

	jointPoly = vcat(getindex.(chains, ranges)...)
	preEnds = [s[1:r[begin]] for (s, r) in zip(chains, ranges)]
	postEnds = [s[r[end]:end] for (s, r) in zip(chains, ranges)]
	groupedEnds = [[reverse(b), e] for (b, e) in zip(preEnds, circshift(postEnds, 1))]

	return jointPoly, groupedEnds
end


"""Compute parametric coordinates (in [0, 1]) for the vertices of a polygonal chain"""
function chainCoords(chain::Vector{Vector{Float64}})::Vector{Float64}
	coords = [0; accumulate(+, [norm(b-a) for (a, b) in partition(chain, 2, 1)])]
	coords /= coords[end]
	return coords
end


"""Interpolate a polygonal chain at a list of coordinates (in [0, 1])"""
function interpolateChain(chain::Vector{Vector{Float64}}, coords::Vector{Float64})::Vector{Vector{Float64}}
	p = chainCoords(chain)
	interp = []
	i = 1
	for (j, c) in enumerate(sort(coords))
		while j > length(interp)
			if p[i] <= c <= p[i+1] 
				push!(interp, ((p[i+1] - c)*chain[i] + (c - p[i])*chain[i+1])/(p[i+1] - p[i]))
			else
				i += 1
			end
		end
	end
	return interp
end


"""Take the parametric mean of two directed polygonal chains"""
function parametricMean(chain1::Vector{Vector{Float64}}, chain2::Vector{Vector{Float64}})::Vector{Vector{Float64}}
	l = max(length(chain1), length(chain2)) #resample chains if necessary
	if length(chain1) < l
		chain1 = interpolateChain(chain1, chainCoords(chain2))
	else
		chain2 = interpolateChain(chain2, chainCoords(chain1))
	end

	return (chain1 .+ chain2)./2
end


"""Consruct a 'tail shaped' TriMesh given a chain and two attachment points"""
function tailMesh(p1::Vector{Float64}, p2::Vector{Float64}, chain::Vector{Vector{Float64}})::TriMesh
	return TriMesh([p1, p2, chain...], [[1, 2, 3], collect.(partition(3:length(chain)+2, 3, 1))...])
end


"""Deform a mesh by harmonic mapping, given a displacement of its boundary nodes"""
function harmonicBoundaryMapping(mesh::TriMesh, boundaryMap::Dict{Vector{Float64}, Vector{Float64}}, krylovDim::Int64)::TriMesh
	boundaryIds = vcat(boundaries(mesh)...)
	Lb = laplacian(mesh, type=:mesh, dirichletNodes=boundaryIds) #mesh laplacian to consider the element shapes, not just connectivity
	bs = zeros(length(mesh.nodes), 3)
	for i in boundaryIds
		bs[i, :] = boundaryMap[mesh.nodes[i]]
	end
	xs = [linsolve(Lb, b, krylovdim=krylovDim)[1] for b in eachcol(bs)] #solve Lb xs = bs
	return TriMesh(collect.(zip(xs...)), mesh.triangles)
end


"""Construct an approximate minimal surface mesh with the skeletonSegments as boundaries, using harmonic mapping"""
function main(skeletonSegmentsPath::String, skeletonMeshPath::String, krylovDim::Int64)
	chains = h5open(skeletonSegmentsPath, "r") do file
		[collect.(eachcol(Float64.(read(file[string(i)])))) for i in 0:4]
	end

	poly, ends = robustJointPoly3D(chains)
	flatPoly = planarProjection(planarLeastSquares(poly), poly)
	polyMesh = harmonicBoundaryMapping(TriMesh(flatPoly), Dict(zip(flatPoly, poly)), krylovDim)
	endMeshes = [tailMesh(e1[1], e2[1], parametricMean(e1, e2)[2:end]) for (e1, e2) in ends]

	save(TriMesh(polyMesh, endMeshes...), skeletonMeshPath)
end


main(ARGS[1:2]..., parse(Int64, ARGS[3]))
