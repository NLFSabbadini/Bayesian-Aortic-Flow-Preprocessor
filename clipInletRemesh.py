import sys
import vtk
import numpy as np
import vmtkTool


def planarLeastSquares(X): 
	"""Compute the least-squares optimal plane for points X = [x1 x2 ... xn]"""
	return np.linalg.solve(np.matmul(X, X.T), np.sum(X, axis=1))


def checkCodirectionality(u, v):
	"""Check the codirectionality of vectos u, v"""
	return np.dot(u, v) > 0


def clipWithPlane(surface, w, towardOrigin=True):
	"""Clip vtkPolyData surface above or below plane w^T x = 1"""
	plane = vtk.vtkPlane()
	plane.SetNormal(w)
	plane.SetOrigin(w/np.dot(w,w))

	clip = vtk.vtkClipPolyData()
	clip.SetInputData(surface)
	clip.SetClipFunction(plane)
	clip.SetInsideOut(not towardOrigin)
	clip.Update()

	return clip.GetOutput()


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


def merge(surfaceList):
	"""Combine all listed vtkPolyData surface objects and merge overlapping vertices"""
	combine = vtk.vtkAppendPolyData()
	for surface in surfaceList:
		combine.AddInputData(surface)
	combine.Update()

	merge = vtk.vtkCleanPolyData()
	merge.SetInputData(combine.GetOutput())
	merge.Update()

	return merge.GetOutput()


def main(surfacePath, inletVectorsPath, outputPath, meshEdgeLength):
	"""Remove connected region upstream of inlet vectors and remesh at desired edge length""" 
	xv = np.load(inletVectorsPath) #load inlet vector positions and values
	p = np.sum(xv[0:3, :], axis=1)/xv.shape[1] #compute mean inlet vector position
	w = planarLeastSquares(xv[0:3, :]) #compute least-squares optimal inlet plane
	d = checkCodirectionality(w, np.sum(xv[3:6, :], axis=1)) #Inlet velocity direction w.r.t. inlet plane normal

	surface = vmtkTool.run("SurfaceReader", InputFileName=surfacePath).Surface #load surface
	surfaceUpstream = clipWithPlane(surface, w, towardOrigin=not d) #get surface in upstream half-space
	surfaceUpstream = removeNearestConnectedRegion(surfaceUpstream, p) #remove connected region nearest to inlet vectors
	surfaceDownstream = clipWithPlane(surface, w, towardOrigin=d) #get surface in downstream half-space
	surface = merge([surfaceUpstream, surfaceDownstream]) #merge maintained regions
	surface = vmtkTool.run("SurfaceRemeshing", #remesh maintained regions at meshEdgeLength
		Surface=surface, ElementSizeMode="edgelength", TargetEdgeLength=meshEdgeLength).Surface
	vmtkTool.run("SurfaceWriter", Surface=surface, OutputFileName=outputPath) #save resulting surface


main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))
