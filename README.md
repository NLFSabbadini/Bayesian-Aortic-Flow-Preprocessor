# Statistical Vascular CFD Pre-Processor
A data processing pipeline for the introduction and propagation to CFD meshes and BCs of measurement uncertainties in vascular geometries and associated inlet flow fields.

## Structure
The pipeline consists of a number of file-in-file-out CLI scripts that can be used independently, which are chained together to form a dependency tree in the 'pipeline' makefile. A number of scripts rely on the Python interface for VMTK/VTK for specific mesh generation/manipulation and computational geometry algorithms, but wherever possible Julia is used. The Python scripts rely on a small convenience module vmtkTool.py that facilitates the use of VMTK, while the Julia scripts rely on the small TriMeshes.jl library for basic manipulation and computational geometry on triangular meshes. A short description of each CLI script is given below.

1. **python-vmtk/clipInletRemesh.py**: Clip the surface mesh so that the inlet aligns with the least-squares plane of the inlet flow vectors and remesh at a desired edge length
2. **julia/graphHarmonics.jl**: Approximate the N lowest frequency graph harmonics of the surface mesh
3. **python-vmtk/skeletonEdges.py**: Compute a wireframe representation of the skeleton of the surface mesh
4. **julia/skeletonMesh.jl**: Mesh the skeleton by approximating the minimal surface of the wireframe
5. **python-vmtk/radialVectors.py**: Compute the 'radial' vectors of the mesh nodes w.r.t. the skeleton
6. **julia/surfaceSamples.jl**: Generate a desired number of smooth meshes, by sampling the joint distribution of harmonic basis weights up to a certain frequency, following from an imposed radial Gaussian distribution of the mesh nodes
7. **python-vmtk/numericalMesh.py**: Generate a tetrahedral volume mesh with flow extensions and boundary layers, given a surface mesh
8. **julia/inletSamples.jl**: Construct and sample the joint distribution of the inlet flow vectors and interpolate the samples onto the inlet boundary of the CFD mesh, using the RBF method

## Files and parameters
For the purpose of interoperability between the CLI scripts, surface meshes are saved in STL and numerical data in HDF5. The final CFD mesh and inlet BC are saved in ANSYS compatible formats. The pipeline script only expects a directory containing the unprocessed surface mesh **rawSurface.stl** and the inlet flow vectors **inletVectors.npy**. Additionally, it takes the following parameters:

- **dir**: Working directory (default: .)
- **procEdgeLength**: Surface processing mesh edge length (default: 0.5)
- **meshEdgeLength**: CFD mesh edge length (default: 0.5)
- **numHarmonics**: Number of graph harmonics to approximate for both surface sampling and skeleton meshing (default: 400)
- **minWavelength**: Minimal surface wavelength/feature size for smoothness (default: 10)
- **radialUncertainty**: Radial uncertainty in surface position (default: 1)
- **inletUncertainty**: 4DFlow MRI uncertainty parameter (default: 1)
- **numSurfaceSamples**: Number of surface meshes to generate (default: 0, i.e. only the mean)
- **numInletSamples**: Number of inlet flow BCs to generate per surface sample (default: 0, i.e. only the mean)
