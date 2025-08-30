import sys
import json
import numpy as np
import vtk
from vmtk import vmtkrenderer
import vmtkTool


def identifyOpenings(surface, openingLabels):
	"""Identify which vessel openings correspond to which labels using a GUI""" 
	seedPoints = vmtkTool.run("BoundaryReferenceSystems", Surface=surface).ReferenceSystems.GetPoints()
	seedPolyData = vtk.vtkPolyData()
	seedPolyData.SetPoints(seedPoints)
	labelsMapper = vtk.vtkLabeledDataMapper();
	labelsMapper.SetInputData(seedPolyData)
	labelsMapper.SetLabelModeToLabelIds()
	labelsActor = vtk.vtkActor2D()
	labelsActor.SetMapper(labelsMapper)

	surfaceMapper = vtk.vtkPolyDataMapper()
	surfaceMapper.SetInputData(surface)
	surfaceMapper.ScalarVisibilityOff()
	surfaceActor = vtk.vtkActor()
	surfaceActor.SetMapper(surfaceMapper)
	surfaceActor.GetProperty().SetOpacity(.25)

	renderer = vmtkrenderer.vmtkRenderer()
	renderer.Initialize()
	renderer.Renderer.AddActor(labelsActor)
	renderer.Renderer.AddActor(surfaceActor)

	openingIds = []
	inputStream = vmtkrenderer.vmtkRendererInputStream(renderer)
	for label in openingLabels:
		inputStream.prompt(f"Please identify {label}: ")
		openingId = int(inputStream.readline())
		if openingId < 0 or openingId >= seedPoints.GetNumberOfPoints():
			raise Exception("Invalid opening id")
		else:
			openingIds.append(openingId)

	return openingIds


def getBoundaryPoints(surface):
	"""Get one point on each boundary of vtkPolyData surface"""
	vtkArray = vmtkTool.run("BoundaryReferenceSystems", Surface=surface).ReferenceSystems.GetPointData().GetArray("Point1")
	return [list(vtkArray.GetTuple(i)) for i in range(vtkArray.GetNumberOfTuples())]


def planarLeastSquares(X): 
	"""Compute the least-squares plane for points X = [x1 x2 ... xn]"""
	return np.linalg.solve(np.matmul(X, X.T), np.sum(X, axis=1))


def cutWithPlane(surface, w):
	"""Cut vtkPolyData with plane w^T x = 1"""
	plane = vtk.vtkPlane()
	plane.SetNormal(w)
	plane.SetOrigin(w/np.dot(w,w))

	clip1 = vtk.vtkClipPolyData()
	clip1.SetInputData(surface)
	clip1.SetClipFunction(plane)
	clip1.Update()

	clip2 = vtk.vtkClipPolyData()
	clip2.SetInputData(surface)
	clip2.SetClipFunction(plane)
	clip2.InsideOutOn()
	clip2.Update()

	combine = vtk.vtkAppendPolyData()
	combine.AddInputData(clip1.GetOutput())
	combine.AddInputData(clip2.GetOutput())
	combine.Update()

	return combine.GetOutput()


def removeNearestConnectedRegion(surface, p):
	"""Remove the connected region of vtkPolyData surface nearest to point p"""
	split = vtk.vtkPolyDataConnectivityFilter() #determine all connected regions
	split.SetInputData(surface)
	split.SetExtractionModeToAllRegions() 
	split.Update()
	
	distances = []
	split.SetExtractionModeToSpecifiedRegions()
	for i in range(split.GetNumberOfExtractedRegions()): #loop over connected region index
		split.AddSpecifiedRegion(i) #extract region i
		split.Update()
		split.DeleteSpecifiedRegion(i)

		dist = vtk.vtkImplicitPolyDataDistance() #compute and save distance from p to region i
		dist.SetInput(split.GetOutput())
		distances.append(dist.EvaluateFunction(p))

	for i in np.argsort(distances)[1:]: #add all regions except nearest to p
		split.AddSpecifiedRegion(i)
	split.Update()

	return split.GetOutput()


def clean(surface):
	"""Clean vtkPolyData surface"""
	cleaner = vtk.vtkCleanPolyData()
	cleaner.SetInputData(surface)
	cleaner.Update()

	return cleaner.GetOutput()


def getBoundaryBarycenters(surface):
	"""Get boundary barycenters of vtkPolyData surface"""
	vtkArray = vmtkTool.run("BoundaryReferenceSystems", Surface=surface).ReferenceSystems.GetPoints().GetData()
	return [list(vtkArray.GetTuple(i)) for i in range(vtkArray.GetNumberOfTuples())]


def main(surfacePath, inletVectorsPath, outputSurfacePath, outputBarycentersPath, meshEdgeLength):
	"""Remove part of inlet upstream of inlet plane defined by inlet vectors and compute the opening barycenters by label""" 
	surface = vmtkTool.run("SurfaceReader", InputFileName=surfacePath).Surface #load surface
	inletVectorPositions = np.load(inletVectorsPath)[0:3, :] #load inlet vector positions

	inletId = identifyOpenings(surface, ["inlet"])[0] #identify the inlet graphically
	inletPoint = getBoundaryPoints(surface)[inletId] #point on inlet edge
	surface = cutWithPlane(surface, planarLeastSquares(inletVectorPositions)) #cut surface with optimal inlet plane
	surface = removeNearestConnectedRegion(surface, inletPoint) #remove upstream part of inlet
	surface = clean(surface) #glue the cutline by merging duplicate vertices

	openingLabels = ["inlet", "branch_1", "branch_2", "branch_3", "outlet"]
	openingIds = identifyOpenings(surface, openingLabels) #identify openings graphically
	openingBarycenters =  getBoundaryBarycenters(surface) #get barycenters
	labeledOpeningBarycenters = {label: openingBarycenters[i] for label, i in zip(openingLabels, openingIds)} #label barycenters

	vmtkTool.run("SurfaceWriter", Surface=surface, OutputFileName=outputSurfacePath) #save clipped surface	
	with open(outputBarycentersPath, 'w') as file: #save labeled barycenters
		json.dump(labeledOpeningBarycenters, file, indent=4)


main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]))
