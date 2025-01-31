import sys
import h5py
import numpy as np
from vmtk import vtkvmtk
import vmtkTool


def planarLeastSquares(X): 
	"""Compute the least-squares optimal plane w^T x = 1 for points X = [x1 x2 ... xn]"""
	sumX = np.sum(X,axis=0)
	return np.linalg.solve(np.matmul(X.T, X), sumX), sumX/X.shape[0] #return plane and origin (centroid of X)


def planarCoords(w, p, xs):
	"""Compute coordinates of 3D point list xs on plane w^T x = 1"""
	P = np.zeros((2, 3))
	P[0,:] = np.cross(w, [0, 0, 1])
	P[0,:] /= np.linalg.norm(P[0,:])
	P[1,:] = np.cross(P[0,:], w/np.linalg.norm(w)) #second axis orthogonal to w and first axis
	return [np.matmul(P, x - p) for x in xs]


def boundaryBarycenters(surface):
	"""Get the barycenters of the surface vtkPolyData borders"""
	boundaries = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor() #extract boundaries
	boundaries.SetInputData(surface)
	boundaries.Update()
	nBoundaries = boundaries.GetOutput().GetNumberOfCells()

	barycenters = [np.zeros(3) for i in range(nBoundaries)]
	for i in range(nBoundaries): #compute barycenters
		vtkvmtk.vtkvmtkBoundaryReferenceSystems.ComputeBoundaryBarycenter(boundaries.GetOutput().GetCell(i).GetPoints(), barycenters[i])

	return barycenters


def cyclicCenterlines(surface, xs):
	"""Compute centerlines cyclically between a list of points xs, given a vtkSurfacePolyData"""
	surface = vmtkTool.run("SurfaceCapper", Surface=surface, Method="centerpoint", Interactive=0).Surface #cap surface
	centerlines = [vmtkTool.run("Centerlines", Surface=surface, SeedSelectorName="pointlist", AppendEndPoints=1,
	SourcePoints=list(a), TargetPoints=list(b)).Centerlines
	for a, b in zip(xs, xs[1:] + [xs[0]])] #compute centerlines between neighboring xs
	
	return centerlines


def smoothCenterlineToConvergence(centerline, tolerance):
	"""Apply Laplace smoothing to centerline vtkPolyData until the maximum update distance reaches tolerance"""
	smoother = vmtkTool.run("CenterlineSmoothing", SmoothingFactor=1, NumberOfSmoothingIterations=1, Centerlines=centerline)
	clold = np.array(centerline.GetPoints().GetData())
	clnew = np.array(smoother.Centerlines.GetPoints().GetData())
	while max(np.sum((clnew - clold)**2, axis=1)) > tolerance**2:
		smoother.Execute()
		clold = clnew
		clnew = np.array(smoother.Centerlines.GetPoints().GetData())
	return clnew #return numpy array


def main(surfacePath, outputPath, centerlineRes):
	"""Compute the centerlines between each pair of neighboring openings (boundaries) and smooth them"""
	surface = vmtkTool.run("SurfaceReader", InputFileName=surfacePath).Surface #load surface
	
	w, p = planarLeastSquares(np.array(surface.GetPoints().GetData())) #compute least-squares optimal plane of surface vertices
	barys = boundaryBarycenters(surface) #compute the surface opening barycenters
	angles = [np.arctan2(y, x) for x, y in planarCoords(w, p, barys)] #compute angles on w plane with origin p
	barys = [barys[i] for i in np.argsort(angles)] #reorder by increasing angle

	centerlines = cyclicCenterlines(surface, barys) #compute the cyclic centerelines between the ordered barycenters
	centerlines = [vmtkTool.run("CenterlineResampling", Centerlines=centerline, Length=centerlineRes).Centerlines for centerline in centerlines] #resample the centerlines
	centerlines = [smoothCenterlineToConvergence(centerline, centerlineRes/100) for centerline in centerlines] #smooth the resampled centerlines
	
	with h5py.File(outputPath, "w") as file: #save centerlines
		for i, centerline in enumerate(centerlines):
			file.create_dataset(str(i), data=centerline)


main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
