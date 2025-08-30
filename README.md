# Bayesian Vascular Flow Preprocessor
A data processing pipeline for the introduction and propagation to CFD meshes and BCs of measurement uncertainties in vascular geometries and associated inlet flow fields.

## Structure
The pipeline consists of a number of file-in-file-out CLI scripts that can be used independently, which are chained together to form a dependency tree in the **pipeline** makefile. A number of scripts rely on the Python interface for VMTK/VTK for specific mesh generation/manipulation and computational geometry algorithms, but wherever possible Julia is used. The Python scripts rely on a small convenience module **vmtkTool.py** that facilitates the use of VMTK, while the Julia scripts rely on the small **TriMeshes.jl** library for basic manipulation and computational geometry on triangular meshes. A short description of each CLI script is given below.

1. **python-vmtk/clipInlet.py**: Clip the surface mesh so that the inlet aligns with the least-squares plane of the inlet flow vectors
2. **julia/surfaceBasis.jl**: Construct a smooth affine basis for the surface mesh, using 10% lowest frequency interior harmonics and the harmonic extension of the planar boundary modes and their harmonic extensions
3. **julia/surfaceDistribution.jl**: Construct a Bayesian posterior distribution for the weights of the smooth affine basis, assuming a diagonal, zero mean Gaussian prior, with power law variance and assuming a Diagonal gaussian likelihood 
4. **julia/surfaceSamples.jl**: Sample from the Bayesian posterior and construct the resulting geometries
5. **python-vmtk/numericalMesh.py**: Generate a tetrahedral volume mesh with flow extensions and boundary layers, given a surface mesh and label the mesh regions consistently
6. **julia/inletSamples.jl**: Construct and sample the posterior distribution of a set of measured inlet flow vectors and interpolate the samples onto the inlet boundary of the CFD mesh, using the masked RBF method to enforce no-slip B.C.s

## Files and parameters
For the purpose of interoperability between the CLI scripts, surface meshes are saved in STL and numerical data in HDF5. The final CFD mesh and inlet BC are saved in ANSYS compatible formats. The pipeline script only expects a directory containing the unprocessed surface mesh **raw.stl** and the inlet flow vectors **inletVectors.npy**. Additionally, it takes the following parameters:

- **dir**: Working directory (default: .)
- **procEdgeLength**: Surface processing mesh edge length (default: 0.5)
- **meshEdgeLength**: CFD mesh edge length (default: 0.5)
- **surfaceUncertainty**: Uncertainty in surface vertex position (default: 0.5)
- **inletUncertainty**: 4DFlow MRI uncertainty parameter (default: 1.0)
- **numSurfaceSamples**: Number of surface meshes to generate (default: 0, i.e. only the mean)
- **numInletSamples**: Number of inlet flow BCs to generate per surface sample (default: 0, i.e. only the mean)
