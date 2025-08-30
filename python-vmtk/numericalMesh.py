import sys
import json
import numpy as np
import vtk
from vmtk import vtkvmtk
import vmtkTool


def vtkArrayToList(vtkArray):
	"""Convert vtkArray to list, using Numpy arrays for vector valued entries"""
	d = vtkArray.GetNumberOfComponents()
	n = vtkArray.GetNumberOfTuples()
	if d == 1:
		return [vtkArray.GetValue(i) for i in range(n)]
	else:
		return [np.array(vtkArray.GetTuple(i)) for i in range(n)]


def getBoundaryData(surface):
	"""Get boundary barycenters, normals and mean radii for vtkPolydata surface"""
	data = vmtkTool.run("BoundaryReferenceSystems", Surface=surface).ReferenceSystems
	barycenters = vtkArrayToList(data.GetPoints().GetData())
	normals = vtkArrayToList(data.GetPointData().GetArray("BoundaryNormals"))
	radii = vtkArrayToList(data.GetPointData().GetArray("BoundaryRadius"))
	return barycenters, normals, radii


def minNormIdx(vectors, offset=0):
	"""Return the minimal norm index of a list of vectors, given an optional offset"""
	return np.argmin([np.linalg.norm(v-offset) for v in vectors])


def addFlowExtensions(surface, outletIds, ratio):
	"""Add flow extensions to specified outlets, with the specified length ratio w.r.t the mean outlet radius"""
	boundaryIds = vtk.vtkIdList()
	for i in outletIds:
		boundaryIds.InsertNextId(i)
	extender = vtkvmtk.vtkvmtkPolyDataFlowExtensionsFilter()
	extender.SetInputData(surface)
	extender.SetExtensionModeToUseNormalToBoundary()
	extender.SetAdaptiveNumberOfBoundaryPoints(1)
	extender.SetExtensionRatio(ratio)
	extender.SetTransitionRatio(ratio)
	extender.SetBoundaryIds(boundaryIds)
	extender.Update()
	newSurface = extender.GetOutput()

	barycenters, normals, radii = getBoundaryData(surface)
	for i in outletIds: #estimate shift in barycenters due to flow extension
		barycenters[i] += normals[i]*radii[i]*ratio
	newBarycenters, _, _ = getBoundaryData(newSurface) #actual shifted barycenters (in different order!)
	newOpeningIndices = [minNormIdx(newBarycenters, offset=barycenter) for barycenter in barycenters] #find new indices based on proximity

	return newSurface, newOpeningIndices


def filterByArrayInterval(dataSet, arrayName, a, b):
	"""Filter vtkDataSet according to vtkArray included in PointData or CellData, selecting values between a and b"""
	selecter = vtk.vtkThreshold()
	selecter.SetInputData(dataSet)
	selecter.SetInputArrayToProcess(0, 0, 0, 1, arrayName)
	selecter.ThresholdBetween(a, b)
	selecter.Update()
	return selecter.GetOutput()


def partitionByIntArray(dataSet, intArrayName, exclude=[]):
	"""Split vtkDataSet according to vktIntArray included in CellData""" 
	ri, rf = dataSet.GetCellData().GetArray(intArrayName).GetRange()
	appender = vtk.vtkAppendFilter()
	for i in range(int(ri), int(rf)+1): #VMTK ships with python 3.6, inline 'if' unsupported
		if i not in exclude: #ignore excluded values
			appender.AddInputData(filterByArrayInterval(dataSet, intArrayName, i, i))
	appender.Update()
	return appender.GetOutput()

	
def nearestCellData(dataSet, cellDataArrayName, point):
	"""Find cell in dataSet nearest to point and return corresponding value from desired cell data array"""
	locator = vtk.vtkCellLocator()
	locator.SetDataSet(dataSet)
	locator.BuildLocator()
	cellId = vtk.reference(0)
	locator.FindClosestPoint(point, [0.0, 0.0, 0.0], cellId, vtk.reference(0), vtk.reference(0.0))
	return dataSet.GetCellData().GetArray(cellDataArrayName).GetValue(cellId)


def extractSurface(dataSet):
	"""Compute surface of vtkDataSet and return as vtkPolyData"""
	surface = vtk.vtkDataSetSurfaceFilter()
	surface.SetInputData(dataSet)
	surface.Update()
	return surface.GetOutput()


def main(surfacePath, barycentersPath, outputVolumeMeshPath, outputInletMeshPath, meshEdgeLength):
	"""Add flow extensions to the Aortic Arch outlets, remesh surface, create volume mesh with boundary layer and extract the inlet boundary mesh"""
	surface = vmtkTool.run("SurfaceReader", InputFileName=surfacePath).Surface #load surface
	with open(barycentersPath, "r") as file:
		labeledBarycenters = json.load(file)
		for l, b in labeledBarycenters.items(): #convert barycenters to numpy vectors
			labeledBarycenters[l] = np.array(b) 

	barycenters, _, _ = getBoundaryData(surface)
	labeledIndices = {l: minNormIdx(barycenters, offset=b) for l, b in labeledBarycenters.items()}
	surface, newIndices = addFlowExtensions(surface, [labeledIndices[l] for l in ["branch_1", "branch_2", "branch_3"]], 10) #add flow extensions to arch branches, with an extension ratio of 10
	barycenters, _, _ = getBoundaryData(surface)
	labeledIndices = {l: newIndices[i] for l, i in labeledIndices.items()}
	labeledBarycenters = {l: barycenters[i] for l, i in labeledIndices.items()}

	mesh = vmtkTool.run("MeshGenerator", #generate volume mesh, surface is remeshed to remove inconsistencies (due to flow extensions here)
		Surface=surface, TargetEdgeLengthFactor=meshEdgeLength, Tetrahedralize=1,
		BoundaryLayer=1, BoundaryLayerOnCaps=0, BoundaryLayerThicknessFactor=1, NumberOfSubLayers=5, SubLayerRatio=0.8333).Mesh
	openingMeshes = partitionByIntArray(mesh, "CellEntityIds", exclude=[0, 1]) #split mesh into regions defined by vmtkMeshGenerator, excluding the internal volume and original surface
	labeledRegions = {l: nearestCellData(openingMeshes, "CellEntityIds", b) for l, b in labeledBarycenters.items()}
	inletMesh = extractSurface(filterByArrayInterval(openingMeshes, "CellEntityIds", labeledRegions["inlet"], labeledRegions["inlet"]))

	vmtkTool.run("SurfaceWriter", Surface=inletMesh, OutputFileName=outputInletMeshPath) #save inlet surface mesh
	vmtkTool.run("MeshWriter", Mesh=mesh, Format="fluent", OutputFileName=outputVolumeMeshPath) #save volume mesh
	with open(outputVolumeMeshPath, "a") as file: #append zone definitions at end of file, will override originals
		file.write(f"(45 (3 wall wall)())\n") #rename vessel wall (always zone 3)
		for l, r in labeledRegions.items(): #rename in/outlet regions to standard names
			file.write(f"(45 ({r+2} wall {l})())\n") #fluent zone numbers are offset by 2


main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]))
