# Bayesian Preprocessing for Vascular CFD
A data processing pipeline for the Bayesian generation of CFD meshes and inlet B.C.s from single measurements of vascular geometries and inlet flow fields.

## Structure
The pipeline consists of a number of file-in-file-out CLI scripts, arranged into a dependency tree in the **pipeline** makefile. Mesh editing and generation tasks are performed using the Python interfaces of VMTK and VTK, while numerical computation tasks related to the statistics are performed using Julia. The Python scripts rely on a small convenience module **vmtkTool.py** that facilitates the use of VMTK, while the Julia scripts rely on the small **TriMeshes.jl** library for basic manipulation and computational geometry of triangular meshes. A short description of each CLI script is given below.

1. **python-vmtk/clipInlet.py**: Clip the surface mesh so that the inlet aligns with the least-squares plane of the inlet flow vectors
2. **julia/surfaceBasis.jl**: Construct a smooth affine basis for the surface mesh, using the 10% lowest frequency interior harmonics and the harmonic extension of the corresponding planar boundary modes
3. **julia/surfaceDistribution.jl**: Construct a Bayesian posterior distribution for the weights of the smooth affine basis, assuming a uniform diagonal Gaussian measurement likelihood and using a data-informed, diagonal, zero mean Gaussian prior, with power-law variance
4. **julia/surfaceSamples.jl**: Sample from the Bayesian posterior and construct the resulting geometries
5. **python-vmtk/numericalMesh.py**: Generate a tetrahedral volume mesh with flow extensions and boundary layers, given a surface mesh and label the mesh regions consistently
6. **julia/inletSamples.jl**: Construct and sample the posterior distribution of a set of measured inlet flow vectors, using a physics-based measurment likelihood and assuming a uniform prior. Then, construct corresponding interpolant functions fulfilling the no-slip B.C.s, using the masked RBF method, and numerically integrate these over the inlet boundary cells of the CFD mesh

## Files and parameters
For the purpose of interoperability between the CLI scripts, surface meshes are saved in STL and numerical data in HDF5 or JSON where human readability is necessary. The final CFD mesh and inlet BC are saved in ANSYS compatible formats. The pipeline script only expects a directory containing the unprocessed surface mesh **raw.stl** and the inlet flow vectors **inletVectors.npy**. Additionally, it takes the following parameters:

- **dir**: Working directory (default: .)
- **procEdgeLength**: Surface processing mesh edge length (default: 0.5)
- **meshEdgeLength**: CFD mesh edge length (default: 0.5)
- **surfaceUncertainty**: Uncertainty in surface vertex position (default: 0.5)
- **inletUncertainty**: 4DFlow MRI uncertainty parameter (default: 1.0)
- **numSurfaceSamples**: Number of surface meshes to generate (default: 0, i.e. only the mean)
- **numInletSamples**: Number of inlet flow BCs to generate per surface sample (default: 0, i.e. only the mean)
