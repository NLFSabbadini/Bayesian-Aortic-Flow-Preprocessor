#Simple module for representing, creating and manipulating triangular meshes in 3D Cartesian space
module TriMeshes


import Gmsh: gmsh
using StaticArrays
using SparseArrays
using LinearAlgebra
using IterTools


export TriMesh, save, areas, centroids, boundaries, linearInterpolation, gaussQuadrature, laplacian


"""Essential data types"""
Node = MVector{3, Float64}
Triangle = MVector{3, UInt64}
AbstractRn = AbstractVector{<:AbstractFloat}
AbstractRns = AbstractVector{<:AbstractRn}


"""Composite type for representing triangular meshes"""
struct TriMesh
	nodes::Vector{Node}
	triangles::Vector{Triangle}

	function TriMesh(nodes, triangles) #Internal constructor including consistency checks
		nodes = convert(Vector{Node}, nodes) #Compatible size & types
		@assert length(nodes) > 2 "found less than 3 nodes"
		@assert all(i -> isassigned(nodes, i), 1:length(nodes)) "found unassigned node"
		@assert	all(node -> all(isfinite, node), nodes) "found NaN and/or Inf in nodes"

		triangles = convert(Vector{Triangle}, triangles) #Compatible size & types
		@assert length(triangles) > 0 "found no triangles"
		@assert all(i -> isassigned(triangles, i), 1:length(triangles)) "found unassigned node"
		@assert all(triangle -> all(triangle .> 0) & all(triangle .<= length(nodes)), triangles) "found out of bounds index in triangles"
		@assert length(unique!(Set.(triangles))) == length(triangles) "found degenerate triangles"
		
		return new(nodes, triangles)
	end
end


"""Construct TriMesh from list of TriMeshes"""
function TriMesh(meshes::TriMesh...)::TriMesh
	nodes = getfield.(meshes, :nodes)
	shifts = (0, accumulate(+, length.(nodes[1:end-1]))...)
	triangles =[[t .+ s for t in m.triangles] for (m, s) in zip(meshes, shifts)]
	return TriMesh(vcat(nodes...), vcat(triangles...))
end


"""Construct TriMesh from file"""
function TriMesh(path::String)::TriMesh
	gmshWasInit = Bool(gmsh.isInitialized())
	gmshWasInit ? nothing : gmsh.initialize()
	gmsh.open(path)
	nodeTags, nodes = gmsh.model.mesh.getNodes()
	triangleTags, triangles = gmsh.model.mesh.getElementsByType(2)
	nodeTagToIndex = Dict(zip(nodeTags,1:length(nodeTags)))
	nodes = collect(partition(nodes, 3))
	triangles = collect(partition(getindex.(Ref(nodeTagToIndex), triangles),3))
	gmshWasInit ? nothing : gmsh.finalize()
	return TriMesh(nodes, triangles)
end


"""Construct TriMesh from planar boundary chain"""
function TriMesh(boundaryChain::T)::TriMesh where T <: AbstractRns
	gmshWasInit = Bool(gmsh.isInitialized())
	gmshWasInit ? nothing : gmsh.initialize()
	gmsh.model.add("fromBoundaryChain")
	for (x, y, z) in boundaryChain
		gmsh.model.geo.addPoint(x, y, z, 10^10)
	end
	for (a, b) in partition([collect(1:length(boundaryChain)); 1], 2, 1)
		gmsh.model.geo.addLine(a, b)
	end
	gmsh.model.geo.addCurveLoop(collect(1:length(boundaryChain)))
	gmsh.model.geo.addPlaneSurface([1])
	gmsh.model.geo.synchronize()
	gmsh.model.mesh.generate(2)

	nodeTags, nodes = gmsh.model.mesh.getNodes()
	triangleTags, triangles = gmsh.model.mesh.getElementsByType(2)
	nodeTagToIndex = Dict(zip(nodeTags,1:length(nodeTags)))
	nodes = collect(partition(nodes, 3))
	triangles = collect(partition(getindex.(Ref(nodeTagToIndex), triangles),3))

	gmsh.model.remove()
	gmshWasInit ? nothing : gmsh.finalize()

	return TriMesh(nodes, triangles)
end


"""Save TriMesh to file"""
function save(mesh::TriMesh, path::String)::Nothing
	gmshWasInit = Bool(gmsh.isInitialized())
	gmshWasInit ? nothing : gmsh.initialize()
	gmsh.model.add(path)
	gmsh.model.addDiscreteEntity(2)
	gmsh.model.mesh.addNodes(2, 1, collect(1:length(mesh.nodes)), reduce(vcat,mesh.nodes))
	gmsh.model.mesh.addElementsByType(1, 2, collect(1:length(mesh.triangles)), reduce(vcat,mesh.triangles))
	gmsh.write(path)
	gmsh.model.remove()
	gmshWasInit ? nothing : gmsh.finalize()
end


"""Compute triangle areas"""
function areas(mesh::TriMesh)::Vector{Float64}
	triAreas = Vector{Float64}(undef, length(mesh.triangles))
	for (i, it) in enumerate(mesh.triangles)
		verts = mesh.nodes[it]
		u = verts[2] - verts[1]
		v = verts[3] - verts[1]
		triAreas[i] = 1/2*sqrt(dot(u,u) + dot(v,v) - dot(u,v)^2)
	end
	return triAreas
end


"""Compute triangle centroids"""
function centroids(mesh::TriMesh)::Vector{Vector{Float64}}
	return [sum(mesh.nodes[ijk])/3 for ijk in mesh.triangles]
end


"""Determine TriMesh boundaries"""
function boundaries(mesh::TriMesh)::Vector{Vector{Int64}}
	numNodes = length(mesh.nodes)
	bGraph = Dict(zip(1:numNodes,[Set{Int}() for _ in 1:numNodes]))

	for (i, j, k) in mesh.triangles
		for (a, b) in ((i,j), (j,k), (k,i))
			if a in bGraph[b]
				pop!(bGraph[b], a)
			else 
				push!(bGraph[a], b)
			end
		end
	end

	for (node, neighbors) in bGraph, neighbor in neighbors
		push!(bGraph[neighbor], node)
	end

	filter!(((node, neighbors),) -> !isempty(neighbors), bGraph)

	bPolys = Vector{Vector{Int}}()

	while length(bGraph) > 0
		start = minimum(keys(bGraph))
		loop = [start, minimum(bGraph[start])]
		while loop[end] != loop[1]
			a, b = bGraph[loop[end]]
			push!(loop, a == loop[end-1] ? b : a)
		end
		push!(bPolys, loop)
		delete!.(Ref(bGraph), loop[2:end])
	end

	return bPolys
end


"""Compute the Laplace matrix for the mesh connectivity (type=:graph) or for the mesh surface (type=:mesh), 
with optional dirichlet nodes and symmetrization for homogeneous B.C.s """
function laplacian(mesh::TriMesh; type::Symbol=:graph, dirichletNodes::Vector{Int64}=Int64[], homogeneous::Bool=false)::SparseMatrixCSC{Float64}
	@assert type in (:graph, :mesh) "type must be :graph or :mesh"

	L = spzeros(length(mesh.nodes), length(mesh.nodes))

	if type == :graph
		for (a, b, c) in mesh.triangles
			for (i,j) in [(a,b), (b,c), (c,a)]
				L[i,j] = -1
				L[j,i] = -1
			end
		end
	elseif type == :mesh
		baryAreas = zeros(length(mesh.nodes))

		for (nodeNums, area) in zip(mesh.triangles, areas(mesh))
			triNodes = getindex.(Ref(mesh.nodes),nodeNums)
			baryAreas[nodeNums] .+= area/3

			for shift in 0:2
				i, j, _ = circshift(nodeNums, shift)
				ni, nj, nk = circshift(triNodes, shift)
				L[i, j] = -dot(ni-nk, nj-nk)/(4*area)
			end
		end

		L += L'
	end

	L -= spdiagm(sum.(eachrow(L)))

	for i in dirichletNodes
		L[i, :] .= 0
		if homogeneous
			L[:, i] .= 0
		end
		L[i, i] = 1
	end

	dropzeros!(L)

	return L
end


"""Linearly interpolate scalar values defined on mesh nodes at arbitrary points on mesh"""
function linearInterpolation(mesh::TriMesh, values::T, points::U)::Vector{Float64} where {T <: AbstractRn, U <: AbstractRns}
	interp = zeros(length(points))

	for iabc in mesh.triangles
		a, b, c = mesh.nodes[iabc]
		u = b - a
		v = c - a
		uu = dot(u, u)
		vv = dot(v, v)
		uv = dot(u, v)
		den = uu*vv - uv^2
		
		for (id, d) in enumerate(points)
			w = d - a
			uw = dot(u, w)
			vw = dot(v, w)

			q = (vv*uw - uv*vw)/den
			r = (uu*vw - uv*uw)/den
			p = 1 - q - r

			if all([p, q, r] .>= 0)
				interp[id] = dot(values[iabc], [p, q, r]) 
			end
		end 
	end

	return interp
end


"""Compute Gauss Quadrature of desired order of function f over the mesh triangles"""
function gaussQuadrature(mesh::TriMesh, f::Function, ::Type{T}; order::Int64=2)::Vector{T} where T <: AbstractArray{<:AbstractFloat}
	gmshWasInit = Bool(gmsh.isInitialized())
	gmshWasInit ? nothing : gmsh.initialize()
	refCoords, refWeights = gmsh.model.mesh.getIntegrationPoints(2, "Gauss$order")
	gmshWasInit ? nothing : gmsh.finalize()

	baryCoords = reshape(refCoords, 3, :)
	baryCoords[3,:] .= 1 .- baryCoords[1,:] .- baryCoords[2,:]

	quads = T[]
	for (indices, area) in zip(mesh.triangles, areas(mesh))
		points = stack(mesh.nodes[indices]) * baryCoords
		push!(quads, 2 * area * sum(f.(eachcol(points)) .* refWeights))
	end

	return quads
end


end
