import sys
import numpy as np
import vtk
from vmtk import vtkvmtk
import vmtkTool


def boundaryBarycenters(surface):
	"""Get the barycenters of the surface vtkPolyData borders"""
	boundaries = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor() #extract boundaries
	boundaries.SetInputData(surface)
	boundaries.Update()
	nBoundaries = boundaries.GetOutput().GetNumberOfCells()

	barycenters = [np.zeros(3) for i in range(nBoundaries)]
	for i in range(nBoundaries): #compute barycenters
		vtkvmtk.vtkvmtkBoundaryReferenceSystems.ComputeBoundaryBarycenter(
			boundaries.GetOutput().GetCell(i).GetPoints(), barycenters[i])

	return barycenters


def addFlowExtensions(outletIds, **args):
	"""Add flow extensions using standard vmtkFlowExtensions arguments and defaults, but to specific outlets"""
	self = vmtkTool.build("FlowExtensions", **args)
	
	boundaryIds = vtk.vtkIdList() #load outletIds (i.e. the vtkvmtkPolyDataBoundaryExtractor Cell number) into a vtkIdList instance
	for outletId in outletIds:
		boundaryIds.InsertNextId(outletId)
	
	vtkExtender = vtkvmtk.vtkvmtkPolyDataFlowExtensionsFilter() #code taken from vmtkFlowExtensions.Execute() for compatibility
	vtkExtender.SetInputData(self.Surface)
	vtkExtender.SetCenterlines(self.Centerlines)
	vtkExtender.SetSigma(self.Sigma)
	vtkExtender.SetAdaptiveExtensionLength(self.AdaptiveExtensionLength)
	vtkExtender.SetAdaptiveExtensionRadius(self.AdaptiveExtensionRadius)
	vtkExtender.SetAdaptiveNumberOfBoundaryPoints(self.AdaptiveNumberOfBoundaryPoints)
	vtkExtender.SetExtensionLength(self.ExtensionLength)
	vtkExtender.SetExtensionRatio(self.ExtensionRatio)
	vtkExtender.SetExtensionRadius(self.ExtensionRadius)
	vtkExtender.SetTransitionRatio(self.TransitionRatio)
	vtkExtender.SetCenterlineNormalEstimationDistanceRatio(self.CenterlineNormalEstimationDistanceRatio)
	vtkExtender.SetNumberOfBoundaryPoints(self.TargetNumberOfBoundaryPoints)
	if self.ExtensionMode == "centerlinedirection":
		vtkExtender.SetExtensionModeToUseCenterlineDirection()
	elif self.ExtensionMode == "boundarynormal":
		vtkExtender.SetExtensionModeToUseNormalToBoundary()
	if self.InterpolationMode == "linear":
		vtkExtender.SetInterpolationModeToLinear()
	elif self.InterpolationMode == "thinplatespline":
		vtkExtender.SetInterpolationModeToThinPlateSpline()
	vtkExtender.SetBoundaryIds(boundaryIds)
	vtkExtender.Update()
	self.Surface = vtkExtender.GetOutput()

	return self.Surface


def splitDataSetByIntArray(dataSet, intArrayName, exclude=[]):
	"""Split vtkDataSet according to vktIntArray included in CellData""" 
	ri, rf = dataSet.GetCellData().GetArray(intArrayName).GetRange()
	appender = vtk.vtkAppendFilter()
	for i in range(int(ri), int(rf)+1): #VMTK ships with python 3.6, inline 'if' unsupported
		if i not in exclude: #ignore excluded values
			selecter = vtk.vtkThreshold()
			selecter.SetInputData(dataSet)
			selecter.SetInputArrayToProcess(0, 0, 0, 1, intArrayName)
			selecter.ThresholdBetween(i, i)
			selecter.Update()
			appender.AddInputData(selecter.GetOutput())
	appender.Update()
	return appender.GetOutput()
	

def nearestConnectedRegion(dataSet, p):
	"""Extract connected region nearest to point p from vtkDataSet"""
	selecter = vtk.vtkConnectivityFilter()
	selecter.SetInputData(dataSet)
	selecter.SetExtractionModeToClosestPointRegion()
	selecter.SetClosestPoint(p)
	selecter.Update()
	return selecter.GetOutput()


def writeSurfaceSTL(dataSet, fileName):
	"""Write surface of vtkDataSet to STL file"""
	surface = vtk.vtkDataSetSurfaceFilter()
	surface.SetInputData(dataSet)
	surface.Update()

	write = vtk.vtkSTLWriter()
	write.SetInputData(surface.GetOutput())
	write.SetFileName(fileName)
	write.Write()


def main(surfacePath, inletVectorsPath, outputVolumeMeshPath, outputInletMeshPath, meshEdgeLength):
	"""Add flow extensions to the Aortic Arch outlets, remesh surface, create volume mesh with boundary layer and extract the inlet boundary mesh"""
	surface = vmtkTool.run("SurfaceReader", InputFileName=surfacePath).Surface #load surface
	archOutlets = np.argsort([z for x, y, z in boundaryBarycenters(surface)])[-3:] #get ids of 3 boundaries with highest barycenter z coordinate
	surface = addFlowExtensions(archOutlets, Surface=surface, ExtensionMode="boundarynormal", AdaptiveExtensionLength=1, ExtensionRatio=10, Interactive=0) #add flow extensions to those
	mesher = vmtkTool.run("MeshGenerator", #generate volume mesh, surface is remeshed to remove inconsistencies (due to flow extensions here)
		Surface=surface, TargetEdgeLengthFactor=meshEdgeLength, Tetrahedralize=1,
		BoundaryLayer=1, BoundaryLayerOnCaps=0, BoundaryLayerThicknessFactor=1, NumberOfSubLayers=5, SubLayerRatio=0.8333)

	volumeMeshSplit = splitDataSetByIntArray(mesher.Mesh, mesher.CellEntityIdsArrayName, exclude=[0, 1]) #split mesh into regions defined by vmtkMeshGenerator, excluding the internal volume and original surface
	inletMesh = nearestConnectedRegion(volumeMeshSplit, np.load(inletVectorsPath)[0:3, 1]) #get contiguous region closest to first inlet vector position

	vmtkTool.run("MeshWriter", Mesh=mesher.Mesh, Format="fluent", OutputFileName=outputVolumeMeshPath) #save volume mesh
	writeSurfaceSTL(inletMesh, outputInletMeshPath) #save inlet surface mesh


main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]))
