import sys
import h5py
import numpy as np
import vmtkTool


def projectSurfaceVerticesToSurface(preImage, imageDomain):
	"""Project vertices of preImage vtkPolyData onto imageDomain vtkPolyData"""
	imageDomain.GetPointData().AddArray(imageDomain.GetPoints().GetData()) #add vertices of imageDomain as PointData to itself
	imageDomainPointsName = imageDomain.GetPoints().GetData().GetName() #get array name for later retrieval
	interpolation = vmtkTool.run("SurfaceProjection", ReferenceSurface=imageDomain, Surface=preImage) #interpolate imageDomain PointData to preImage based on minimal distance
	return np.array(interpolation.Surface.GetPointData().GetArray(imageDomainPointsName)) #get interpolated imageDomain vertices (closest imageDomain points!)


def main(surface1Path, surface2Path, outputPath):
	"""Project vertices of surface1 onto surface2 and compute the relative normals/radii"""
	surface1 = vmtkTool.run("SurfaceReader", InputFileName=surface1Path).Surface #load surface1
	surface2 = vmtkTool.run("SurfaceReader", InputFileName=surface2Path).Surface #load surface2
	projectedVertices = projectSurfaceVerticesToSurface(surface1, surface2) #project vertices
	vectors = np.array(surface1.GetPoints().GetData()) - np.array(projectedVertices)
	radii = np.linalg.norm(vectors, axis=1)
	normals = vectors / radii[:, np.newaxis]

	with h5py.File(outputPath, "w") as file: #save resulting array
		file.create_dataset("normals", data=normals)
		file.create_dataset("radii", data=radii)


main(sys.argv[1], sys.argv[2], sys.argv[3])
