// ============================================================================
// CHRONOS STANDARD LIBRARY — PART 3
// Simulation, Data Science, Data Engineering, Fine-Grained Memory Management
// ============================================================================
// This covers:
//   - Multi-scale simulation (astrophysics, fluid dynamics, FEM, CAD, etc.)
//   - Plotting, charting, 3D visualization
//   - Complete data science toolkit
//   - Data engineering, scraping, ETL pipelines
//   - Extremely fine-grained manual memory management
// ============================================================================

use std::collections::HashMap;
use std::time::Duration;

// ============================================================================
// SECTION 1: MULTI-SCALE SIMULATION ENGINE
// ============================================================================
// Chronos treats simulation as a first-class language feature. The simulation
// engine supports every scale from quantum mechanics to astrophysics, and
// every domain from fluid dynamics to structural engineering.
//
// The key insight: all simulations share a common mathematical structure —
// they solve PDEs/ODEs on a domain with boundary conditions. Chronos
// provides a unified interface for this, with specialized backends for
// each domain.

/// A simulation domain — the "world" being simulated.
#[derive(Debug, Clone)]
pub struct SimulationDomain {
    pub name: String,
    pub dimensionality: u8,              // 1D, 2D, 3D, 4D (spacetime)
    pub geometry: DomainGeometry,
    pub mesh: Option<SimulationMesh>,
    pub boundary_conditions: Vec<BoundaryCondition>,
    pub initial_conditions: Vec<InitialCondition>,
    pub time_config: Option<TimeConfig>,
    pub physics: Vec<PhysicsModel>,
    pub solver: SolverConfig,
    pub output: OutputConfig,
    pub parallelism: ParallelismConfig,
}

/// Geometry of the simulation domain.
#[derive(Debug, Clone)]
pub enum DomainGeometry {
    // --- Simple geometries ---
    Line { length: f64 },
    Rectangle { width: f64, height: f64 },
    Box3D { width: f64, height: f64, depth: f64 },
    Circle { radius: f64 },
    Sphere { radius: f64 },
    Cylinder { radius: f64, height: f64 },
    Torus { major_radius: f64, minor_radius: f64 },
    
    // --- CAD-imported geometry ---
    STLFile(String),                     // STL mesh file
    STEPFile(String),                    // STEP/IGES CAD format
    OBJFile(String),
    PointCloud(Vec<[f64; 3]>),
    
    // --- Astrophysics-scale ---
    StellarSystem {
        bodies: Vec<CelestialBody>,
        scale: CosmicScale,
    },
    CosmologicalVolume {
        size_megaparsecs: f64,
        expansion_model: CosmologyModel,
    },
    
    // --- Procedural ---
    Implicit { sdf: String },            // Signed distance function expression
    Parametric { expr: String, params: Vec<(String, f64, f64)> },
    Fractal { kind: FractalKind, iterations: u32 },
    
    // --- Composite ---
    Union(Vec<DomainGeometry>),
    Intersection(Vec<DomainGeometry>),
    Difference(Box<DomainGeometry>, Box<DomainGeometry>),
}

#[derive(Debug, Clone)]
pub struct CelestialBody {
    pub name: String,
    pub mass_kg: f64,
    pub radius_m: f64,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub body_type: CelestialType,
}

#[derive(Debug, Clone)]
pub enum CelestialType {
    Star { luminosity_solar: f64, spectral_class: String },
    Planet { is_gas_giant: bool },
    Moon, Asteroid, Comet, BlackHole { spin: f64 },
    NeutronStar { magnetic_field_tesla: f64 },
    WhiteDwarf, GalaxyCenter,
}

#[derive(Debug, Clone)]
pub enum CosmicScale {
    Planetary,           // ~10^6 m
    SolarSystem,         // ~10^13 m
    Stellar,             // ~10^16 m (parsecs)
    Galactic,            // ~10^21 m (kiloparsecs)
    Intergalactic,       // ~10^24 m (megaparsecs)
    Observable,          // ~10^27 m (gigaparsecs)
}

#[derive(Debug, Clone)]
pub enum CosmologyModel {
    LCDM { h0: f64, omega_m: f64, omega_lambda: f64 },
    Einstein_deSitter,
    Milne,
    Custom(String),      // Custom Friedmann equation
}

#[derive(Debug, Clone)]
pub enum FractalKind {
    Mandelbrot, Julia(f64, f64), MengerSponge, SierpinskiTriangle,
    KochSnowflake, DragonCurve, BurningShip,
}

/// The simulation mesh — discretization of the domain.
#[derive(Debug, Clone)]
pub struct SimulationMesh {
    pub kind: MeshKind,
    pub refinement: MeshRefinement,
    pub quality: MeshQuality,
}

#[derive(Debug, Clone)]
pub enum MeshKind {
    // --- Structured ---
    CartesianGrid { cells: Vec<usize> },
    CurvilinearGrid,
    PolarGrid { nr: usize, ntheta: usize },
    SphericalGrid { nr: usize, ntheta: usize, nphi: usize },
    
    // --- Unstructured ---
    TriangularMesh,
    TetrahedralMesh,
    HexahedralMesh,
    PolyhedralMesh,
    MixedElementMesh,
    
    // --- Particle-based ---
    SPH { particle_count: usize, kernel: SPHKernel },
    DEM { particle_count: usize },        // Discrete Element Method
    MPS,                                   // Moving Particle Semi-implicit
    
    // --- Adaptive ---
    AMR { max_levels: u32 },              // Adaptive Mesh Refinement
    Octree { max_depth: u32 },
    
    // --- Meshless ---
    RBF,                                   // Radial Basis Functions
    RKPM,                                  // Reproducing Kernel Particle Method
    ElementFree,
}

#[derive(Debug, Clone)]
pub enum SPHKernel { CubicSpline, Wendland, Gaussian }

#[derive(Debug, Clone)]
pub enum MeshRefinement {
    Uniform,
    Adaptive { error_threshold: f64 },
    GradedToward { point: Vec<f64>, min_size: f64 },
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct MeshQuality {
    pub min_element_size: f64,
    pub max_element_size: f64,
    pub target_quality: f64,        // 0.0 = worst, 1.0 = perfect
    pub smoothing_iterations: u32,
}

/// Physics models — what equations govern the simulation.
#[derive(Debug, Clone)]
pub enum PhysicsModel {
    // === MECHANICS ===
    NewtonianGravity,
    GeneralRelativity { metric: String },
    RigidBodyDynamics,
    ElasticSolid { model: ElasticModel },
    PlasticSolid { model: PlasticityModel },
    ViscoplasticSolid,
    Fracture { model: FractureModel },
    ContactMechanics { friction: FrictionModel },
    Vibration { damping: f64 },
    
    // === FLUID DYNAMICS ===
    IncompressibleNavierStokes { reynolds: f64 },
    CompressibleNavierStokes { mach: f64 },
    Euler,                               // Inviscid compressible flow
    StokesFlow,                          // Low Reynolds number
    ShallowWater,
    Boussinesq,
    BoundaryLayer { turbulence: TurbulenceModel },
    MultiphaseFlow { model: MultiphaseModel },
    FreeSurface { method: FreeSurfaceMethod },
    Porous { model: PorousMediaModel },
    NonNewtonian { model: RheologyModel },
    
    // === THERMODYNAMICS ===
    HeatConduction,
    Convection,
    Radiation { model: RadiationModel },
    PhaseChange { model: PhaseChangeModel },
    Combustion { model: CombustionModel },
    
    // === ELECTROMAGNETICS ===
    Maxwell,
    Electrostatics,
    Magnetostatics,
    EddyCurrents,
    WavePropagation { medium: String },
    Plasma { model: PlasmaModel },
    
    // === QUANTUM ===
    Schrodinger,
    DFT,                                 // Density Functional Theory
    MolecularDynamics { potential: MDPotential },
    MonteCarloQuantum,
    TightBinding,
    HartreeFock,
    
    // === ASTROPHYSICS ===
    NBody { integrator: NBodyIntegrator },
    StellarEvolution,
    AccretionDisk,
    MHD,                                 // Magnetohydrodynamics
    CosmologicalNBody,
    RadiativeTransfer,
    GravitationalWaves,
    
    // === STRUCTURAL ANALYSIS (FEM) ===
    LinearStatic,
    NonlinearStatic,
    ModalAnalysis,
    BucklingAnalysis,
    TransientDynamic,
    HarmonicResponse,
    RandomVibration,
    SeismicAnalysis { spectrum: Vec<(f64, f64)> },
    Fatigue { model: FatigueModel },
    Creep { model: CreepModel },
    Composite { layup: Vec<CompositeLayer> },
    
    // === MULTIPHYSICS ===
    FluidStructureInteraction,
    ThermoMechanical,
    ElectroMechanical,
    PiezoElectric,
    ThermoElectric,
    ChemicalReaction { reactions: Vec<ChemicalReaction> },
    Acoustics,
    Aeroacoustics,
    
    // === WEATHER & CLIMATE ===
    AtmosphericGCM,                      // General Circulation Model
    OceanGCM,
    WeatherPrediction { model: WeatherModel },
    ClimateProjection,
    
    // === BIOLOGICAL ===
    PopulationDynamics { model: String },
    Epidemiological { model: EpiModel },
    NeuralSimulation { model: NeuralModel },
    CellularAutomaton { rule: String },
    AgentBased { agents: usize },
    
    // === CUSTOM PDE ===
    CustomPDE {
        equation: String,                // Symbolic PDE expression
        variables: Vec<String>,
        parameters: HashMap<String, f64>,
    },
}

// --- Sub-model enums (abbreviated for space; each would have many variants) ---
#[derive(Debug, Clone)]
pub enum ElasticModel { LinearIsotropic { E: f64, nu: f64 }, Orthotropic(Vec<f64>), Hyperelastic(String), NeoHookean, MooneyRivlin }
#[derive(Debug, Clone)]
pub enum PlasticityModel { VonMises { yield_stress: f64 }, Tresca, DruckerPrager, CamClay, JohnsonCook }
#[derive(Debug, Clone)]
pub enum FractureModel { LinearFracture, Cohesive, XFEM, PhaseField, Peridynamics }
#[derive(Debug, Clone)]
pub enum FrictionModel { Coulomb(f64), Penalty(f64), AugmentedLagrangian }
#[derive(Debug, Clone)]
pub enum TurbulenceModel { Laminar, KOmega, KEpsilon, SpalartAllmaras, LES(LESModel), DNS, RANS, DES, WMLES }
#[derive(Debug, Clone)]
pub enum LESModel { Smagorinsky, DynamicSmagorinsky, WALE, Sigma }
#[derive(Debug, Clone)]
pub enum MultiphaseModel { VOF, LevelSet, EulerianEulerian, EulerianLagrangian, MixtureModel }
#[derive(Debug, Clone)]
pub enum FreeSurfaceMethod { VOF, LevelSet, SPH, MarkerAndCell }
#[derive(Debug, Clone)]
pub enum PorousMediaModel { Darcy, DarcyForchheimer, BrinkmanForchheimer }
#[derive(Debug, Clone)]
pub enum RheologyModel { PowerLaw { n: f64, K: f64 }, Bingham { yield_stress: f64 }, HerschelBulkley, CarreauYasuda }
#[derive(Debug, Clone)]
pub enum RadiationModel { P1, DiscreteOrdinates, MonteCarlo, RosselandDiffusion }
#[derive(Debug, Clone)]
pub enum PhaseChangeModel { Solidification, Boiling, Evaporation, Sublimation }
#[derive(Debug, Clone)]
pub enum CombustionModel { EddyDissipation, FiniteRate, FlameletModel, PDF }
#[derive(Debug, Clone)]
pub enum PlasmaModel { PIC, Fluid, Hybrid, Gyrokinetic }
#[derive(Debug, Clone)]
pub enum MDPotential { LennardJones, EAM, MEAM, Tersoff, ReaxFF, CustomForceField(String) }
#[derive(Debug, Clone)]
pub enum NBodyIntegrator { Leapfrog, Hermite, RungeKutta4, BarnesHut, FMM, Symplectic }
#[derive(Debug, Clone)]
pub enum FatigueModel { SN, StrainLife, CrackGrowth, Dang_Van, Critical_Plane }
#[derive(Debug, Clone)]
pub enum CreepModel { Norton, Power_Law, Nabarro_Herring, Coble }
#[derive(Debug, Clone)]
pub struct CompositeLayer { pub material: String, pub thickness: f64, pub angle: f64 }
#[derive(Debug, Clone)]
pub struct ChemicalReaction { pub reactants: Vec<(String, f64)>, pub products: Vec<(String, f64)>, pub rate: f64 }
#[derive(Debug, Clone)]
pub enum WeatherModel { GFS, WRF, ECMWF, MPAS, FV3 }
#[derive(Debug, Clone)]
pub enum EpiModel { SIR, SEIR, SEIRS, SIS, AgentBased }
#[derive(Debug, Clone)]
pub enum NeuralModel { HodgkinHuxley, IntegrateAndFire, FitzHughNagumo, Izhikevich }

/// Boundary conditions.
#[derive(Debug, Clone)]
pub enum BoundaryCondition {
    Dirichlet { boundary: String, value: String },        // Fixed value
    Neumann { boundary: String, flux: String },           // Fixed gradient
    Robin { boundary: String, alpha: f64, beta: f64, gamma: String }, // Mixed
    Periodic { boundary_a: String, boundary_b: String },
    Symmetry { boundary: String },
    FarField { boundary: String, mach: f64, aoa: f64 },
    Wall { boundary: String, no_slip: bool },
    Inflow { boundary: String, profile: String },
    Outflow { boundary: String, backpressure: Option<f64> },
    Absorbing { boundary: String },                        // PML for waves
    Radiation { boundary: String, emissivity: f64, ambient_temp: f64 },
}

/// Initial conditions.
#[derive(Debug, Clone)]
pub struct InitialCondition {
    pub variable: String,
    pub expression: String,              // Symbolic expression over spatial coords
}

/// Time integration configuration.
#[derive(Debug, Clone)]
pub struct TimeConfig {
    pub start: f64,
    pub end: f64,
    pub dt: TimeStep,
    pub method: TimeIntegration,
}

#[derive(Debug, Clone)]
pub enum TimeStep {
    Fixed(f64),
    Adaptive { min: f64, max: f64, tolerance: f64 },
    CFL { number: f64 },                // Courant-Friedrichs-Lewy condition
}

#[derive(Debug, Clone)]
pub enum TimeIntegration {
    ForwardEuler, BackwardEuler, CrankNicolson,
    RungeKutta(u8), SSPRK(u8),          // Strong Stability Preserving RK
    BDF(u8), Newmark { beta: f64, gamma: f64 },
    HHT { alpha: f64 }, GeneralizedAlpha { rho_inf: f64 },
}

/// Solver configuration.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    pub linear_solver: LinearSolver,
    pub nonlinear_solver: Option<NonlinearSolver>,
    pub preconditioner: Option<Preconditioner>,
    pub tolerance: f64,
    pub max_iterations: u32,
}

#[derive(Debug, Clone)]
pub enum LinearSolver {
    Direct(DirectSolver),
    Iterative(IterativeSolver),
}

#[derive(Debug, Clone)]
pub enum DirectSolver { LU, Cholesky, LDLT, Pardiso, MUMPS, SuperLU, UMFPACK }

#[derive(Debug, Clone)]
pub enum IterativeSolver {
    CG, BiCG, BiCGSTAB, GMRES { restart: u32 }, MINRES,
    Multigrid { levels: u32, smoother: String },
    AMG,                                 // Algebraic Multigrid
}

#[derive(Debug, Clone)]
pub enum NonlinearSolver {
    NewtonRaphson, ModifiedNewton, BFGS, LBFGS,
    TrustRegion, LineSearch, ArcLength, LoadControl,
}

#[derive(Debug, Clone)]
pub enum Preconditioner {
    Jacobi, GaussSeidel, SOR(f64), ILU(u32), ICC(u32),
    AMG, BlockJacobi, AdditiveSchwarz,
}

/// Output configuration.
#[derive(Debug, Clone)]
pub struct OutputConfig {
    pub format: OutputFormat,
    pub frequency: OutputFrequency,
    pub variables: Vec<String>,
    pub derived_quantities: Vec<DerivedQuantity>,
}

#[derive(Debug, Clone)]
pub enum OutputFormat { VTK, HDF5, CGNS, Ensight, Exodus, CSV, ParaView, Tecplot, NetCDF }

#[derive(Debug, Clone)]
pub enum OutputFrequency { EveryStep, EveryNSteps(u32), EveryDt(f64), AtTimes(Vec<f64>) }

#[derive(Debug, Clone)]
pub enum DerivedQuantity {
    VonMisesStress, PrincipalStresses, StrainEnergy,
    Vorticity, QCriterion, WallShearStress, LiftCoefficient, DragCoefficient,
    NusseltNumber, Pressure, Temperature, Velocity, Density,
    KineticEnergy, PotentialEnergy, TotalEnergy, Entropy,
    Custom { name: String, expression: String },
}

/// Parallelism configuration.
#[derive(Debug, Clone)]
pub struct ParallelismConfig {
    pub strategy: ParallelStrategy,
    pub domain_decomposition: DecompositionMethod,
    pub load_balancing: LoadBalancing,
}

#[derive(Debug, Clone)]
pub enum ParallelStrategy { Serial, OpenMP(u32), MPI(u32), Hybrid { mpi: u32, omp: u32 }, GPU, MultiGPU(u32) }

#[derive(Debug, Clone)]
pub enum DecompositionMethod { Metis, Scotch, RCB, SFC, Manual(Vec<usize>) }

#[derive(Debug, Clone)]
pub enum LoadBalancing { Static, Dynamic, Adaptive }

// ============================================================================
// SECTION 2: DATA VISUALIZATION (Plotting, Charts, 3D)
// ============================================================================

/// A plot specification. The compiler can render these to SVG, PNG, PDF,
/// or interactive WebGL.
#[derive(Debug, Clone)]
pub enum PlotKind {
    // --- 2D plots ---
    Line { x: Vec<f64>, y: Vec<f64>, style: LineStyle },
    Scatter { x: Vec<f64>, y: Vec<f64>, sizes: Option<Vec<f64>>, colors: Option<Vec<f64>> },
    Bar { categories: Vec<String>, values: Vec<f64>, orientation: Orientation },
    Histogram { data: Vec<f64>, bins: BinSpec },
    Pie { labels: Vec<String>, values: Vec<f64> },
    Area { x: Vec<f64>, y: Vec<f64>, stacked: bool },
    BoxPlot { data: Vec<Vec<f64>>, labels: Vec<String> },
    Violin { data: Vec<Vec<f64>>, labels: Vec<String> },
    Heatmap { data: Vec<Vec<f64>>, x_labels: Vec<String>, y_labels: Vec<String> },
    Contour { x: Vec<f64>, y: Vec<f64>, z: Vec<Vec<f64>>, filled: bool },
    StreamPlot { x: Vec<f64>, y: Vec<f64>, u: Vec<Vec<f64>>, v: Vec<Vec<f64>> },
    Quiver { x: Vec<f64>, y: Vec<f64>, u: Vec<f64>, v: Vec<f64> },
    Polar { r: Vec<f64>, theta: Vec<f64> },
    ErrorBar { x: Vec<f64>, y: Vec<f64>, yerr: Vec<f64>, xerr: Option<Vec<f64>> },
    Stem { x: Vec<f64>, y: Vec<f64> },
    StepPlot { x: Vec<f64>, y: Vec<f64> },
    Waterfall { categories: Vec<String>, values: Vec<f64> },
    Funnel { stages: Vec<String>, values: Vec<f64> },
    Radar { axes: Vec<String>, values: Vec<Vec<f64>> },
    Sankey { nodes: Vec<String>, links: Vec<(usize, usize, f64)> },
    Treemap { labels: Vec<String>, values: Vec<f64>, parents: Vec<Option<usize>> },
    Sunburst { labels: Vec<String>, values: Vec<f64>, parents: Vec<Option<usize>> },
    Candlestick { dates: Vec<String>, open: Vec<f64>, high: Vec<f64>, low: Vec<f64>, close: Vec<f64> },
    Gantt { tasks: Vec<GanttTask> },
    
    // --- 3D plots ---
    Surface3D { x: Vec<f64>, y: Vec<f64>, z: Vec<Vec<f64>> },
    Scatter3D { x: Vec<f64>, y: Vec<f64>, z: Vec<f64> },
    Line3D { x: Vec<f64>, y: Vec<f64>, z: Vec<f64> },
    Wireframe3D { x: Vec<f64>, y: Vec<f64>, z: Vec<Vec<f64>> },
    Mesh3D { vertices: Vec<[f64; 3]>, faces: Vec<[usize; 3]>, values: Option<Vec<f64>> },
    Isosurface { data: Vec<Vec<Vec<f64>>>, iso_value: f64 },
    VolumeRendering { data: Vec<Vec<Vec<f64>>>, transfer_fn: TransferFunction },
    PointCloud3D { points: Vec<[f64; 3]>, colors: Option<Vec<[f64; 3]>> },
    VectorField3D { origins: Vec<[f64; 3]>, vectors: Vec<[f64; 3]> },
    
    // --- Statistical ---
    QQPlot { data: Vec<f64>, distribution: String },
    ACFPlot { data: Vec<f64>, lags: usize },
    PACFPlot { data: Vec<f64>, lags: usize },
    PairPlot { data: Vec<Vec<f64>>, labels: Vec<String> },
    CorrelationMatrix { data: Vec<Vec<f64>>, labels: Vec<String> },
    ROCCurve { fpr: Vec<f64>, tpr: Vec<f64>, auc: f64 },
    ConfusionMatrix { matrix: Vec<Vec<usize>>, labels: Vec<String> },
    KaplanMeier { times: Vec<f64>, events: Vec<bool> },
    
    // --- Network / Graph ---
    NetworkGraph { nodes: Vec<NodeVis>, edges: Vec<EdgeVis>, layout: GraphLayout },
    Dendrogram { data: Vec<Vec<f64>>, linkage: String },
    
    // --- Maps ---
    Choropleth { regions: Vec<String>, values: Vec<f64>, geojson: String },
    ScatterMap { lats: Vec<f64>, lons: Vec<f64>, values: Option<Vec<f64>> },
    HeatmapMap { lats: Vec<f64>, lons: Vec<f64>, weights: Vec<f64> },
    FlowMap { origins: Vec<(f64, f64)>, destinations: Vec<(f64, f64)>, values: Vec<f64> },
    
    // --- Composite ---
    Subplot { rows: u32, cols: u32, plots: Vec<(u32, u32, Box<PlotKind>)> },
    Overlay(Vec<PlotKind>),
    Animation { frames: Vec<PlotKind>, interval_ms: u32 },
    Interactive { base: Box<PlotKind>, widgets: Vec<Widget> },
}

#[derive(Debug, Clone)]
pub struct LineStyle { pub color: String, pub width: f64, pub dash: Option<String>, pub marker: Option<String> }
#[derive(Debug, Clone)]
pub enum Orientation { Horizontal, Vertical }
#[derive(Debug, Clone)]
pub enum BinSpec { Count(usize), Width(f64), Edges(Vec<f64>), Auto }
#[derive(Debug, Clone)]
pub struct GanttTask { pub name: String, pub start: f64, pub duration: f64, pub deps: Vec<String> }
#[derive(Debug, Clone)]
pub struct TransferFunction { pub breakpoints: Vec<(f64, [f64; 4])> }
#[derive(Debug, Clone)]
pub struct NodeVis { pub id: usize, pub label: String, pub size: f64, pub color: String }
#[derive(Debug, Clone)]
pub struct EdgeVis { pub from: usize, pub to: usize, pub weight: f64 }
#[derive(Debug, Clone)]
pub enum GraphLayout { ForceDirected, Circular, Hierarchical, Spring, Spectral, Kamada_Kawai }
#[derive(Debug, Clone)]
pub enum Widget { Slider { label: String, min: f64, max: f64 }, Dropdown { label: String, options: Vec<String> }, Toggle(String) }


// ============================================================================
// SECTION 3: DATA SCIENCE TOOLKIT
// ============================================================================
// Everything needed for an extremely advanced data science project.

/// A DataFrame — the fundamental data structure for data science.
/// Column-oriented for cache efficiency, with lazy evaluation.
#[derive(Debug, Clone)]
pub struct DataFrame {
    pub columns: Vec<Column>,
    pub row_count: usize,
    pub schema: Schema,
    pub lazy_ops: Vec<LazyOp>,          // Operations deferred until materialization
}

#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub dtype: DataType,
    pub data: ColumnData,
    pub null_bitmap: Option<Vec<u64>>,   // Bitwise null tracking (Arrow-style)
}

#[derive(Debug, Clone)]
pub enum DataType {
    Boolean, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
    Float32, Float64, Decimal(u8, u8),
    Utf8, LargeUtf8, Binary, LargeBinary,
    Date32, Date64, Timestamp(TimeUnit), Duration(TimeUnit), Interval(IntervalUnit),
    List(Box<DataType>), FixedSizeList(Box<DataType>, usize),
    Struct(Vec<(String, DataType)>), Map(Box<DataType>, Box<DataType>),
    Dictionary(Box<DataType>, Box<DataType>),
    Categorical(Vec<String>),
    Embedding(usize),                    // ML embedding vector
}

#[derive(Debug, Clone)]
pub enum TimeUnit { Second, Millisecond, Microsecond, Nanosecond }
#[derive(Debug, Clone)]
pub enum IntervalUnit { YearMonth, DayTime, MonthDayNano }
#[derive(Debug, Clone)]
pub enum ColumnData { Boolean(Vec<bool>), Int64(Vec<i64>), Float64(Vec<f64>), Utf8(Vec<String>), Binary(Vec<Vec<u8>>), Empty }

/// Lazy operations — computed only when the result is needed.
#[derive(Debug, Clone)]
pub enum LazyOp {
    Select(Vec<String>),
    Filter(FilterExpr),
    GroupBy { keys: Vec<String>, aggs: Vec<Aggregation> },
    Join { other: Box<DataFrame>, on: Vec<(String, String)>, how: JoinKind },
    Sort { by: Vec<String>, ascending: Vec<bool> },
    Limit(usize),
    Offset(usize),
    Distinct(Vec<String>),
    WithColumn { name: String, expr: ColumnExpr },
    Rename(Vec<(String, String)>),
    Drop(Vec<String>),
    Pivot { index: String, columns: String, values: String, agg: AggKind },
    Unpivot { id_vars: Vec<String>, value_vars: Vec<String> },
    Window { partition_by: Vec<String>, order_by: Vec<String>, expr: WindowExpr },
    Explode(String),
    FillNull { column: String, strategy: FillStrategy },
    Cast { column: String, to: DataType },
    Sample { n: Option<usize>, fraction: Option<f64>, seed: Option<u64> },
    CrossJoin(Box<DataFrame>),
}

#[derive(Debug, Clone)]
pub enum FilterExpr { Eq(String, String), Gt(String, f64), Lt(String, f64), IsNull(String), In(String, Vec<String>), And(Box<FilterExpr>, Box<FilterExpr>), Or(Box<FilterExpr>, Box<FilterExpr>), Not(Box<FilterExpr>), Regex(String, String), Between(String, f64, f64), Custom(String) }
#[derive(Debug, Clone)]
pub enum Aggregation { Count, Sum(String), Mean(String), Median(String), Min(String), Max(String), Std(String), Var(String), First(String), Last(String), Quantile(String, f64), CountDistinct(String), Collect(String), Custom { name: String, expr: String } }
#[derive(Debug, Clone)]
pub enum AggKind { Sum, Mean, Count, Min, Max, First, Last }
#[derive(Debug, Clone)]
pub enum JoinKind { Inner, Left, Right, Full, Cross, Semi, Anti, AsOf { tolerance: f64 } }
#[derive(Debug, Clone)]
pub enum ColumnExpr { Literal(String), Column(String), BinaryOp(Box<ColumnExpr>, String, Box<ColumnExpr>), Function(String, Vec<ColumnExpr>), Conditional { when: Box<FilterExpr>, then: Box<ColumnExpr>, otherwise: Box<ColumnExpr> } }
#[derive(Debug, Clone)]
pub enum WindowExpr { RowNumber, Rank, DenseRank, Lag(String, usize), Lead(String, usize), RunningSum(String), RunningMean(String), CumSum(String), Ntile(usize), FirstValue(String), LastValue(String) }
#[derive(Debug, Clone)]
pub enum FillStrategy { Value(String), Forward, Backward, Mean, Median, Interpolate }

/// Schema describes the structure of a DataFrame.
#[derive(Debug, Clone)]
pub struct Schema {
    pub fields: Vec<(String, DataType)>,
}

/// Machine Learning models — built into the language.
#[derive(Debug, Clone)]
pub enum MLModel {
    // --- Supervised: Regression ---
    LinearRegression, Ridge(f64), Lasso(f64), ElasticNet(f64, f64),
    PolynomialRegression(u32), SVR { kernel: Kernel },
    DecisionTreeRegressor { max_depth: Option<u32> },
    RandomForestRegressor { n_trees: u32, max_depth: Option<u32> },
    GradientBoostingRegressor { n_estimators: u32, learning_rate: f64 },
    XGBoost(XGBoostParams), LightGBM(LGBMParams), CatBoost(CatBoostParams),
    GaussianProcess { kernel: GPKernel },
    
    // --- Supervised: Classification ---
    LogisticRegression, SVM { kernel: Kernel, C: f64 },
    DecisionTreeClassifier { max_depth: Option<u32> },
    RandomForestClassifier { n_trees: u32 },
    GradientBoostingClassifier { n_estimators: u32, learning_rate: f64 },
    NaiveBayes(NBKind), KNN(u32),
    
    // --- Unsupervised: Clustering ---
    KMeans(u32), DBSCAN { eps: f64, min_samples: usize },
    HierarchicalClustering { linkage: String },
    GaussianMixture { n_components: u32 }, OPTICS, MeanShift,
    SpectralClustering { n_clusters: u32 }, HDBSCAN,
    
    // --- Unsupervised: Dimensionality Reduction ---
    PCA(u32), KernelPCA(u32, Kernel), TruncatedSVD(u32),
    TSNE { perplexity: f64 }, UMAP { n_neighbors: u32, min_dist: f64 },
    Isomap(u32), LLE(u32), FactorAnalysis(u32),
    NMF(u32), ICA(u32), Autoencoder(Vec<usize>),
    
    // --- Time Series ---
    ARIMA(u32, u32, u32), SARIMA(u32, u32, u32, u32, u32, u32, u32),
    ExponentialSmoothing(ESKind), Prophet, VAR(u32),
    LSTM { hidden_size: usize, layers: usize },
    Transformer { d_model: usize, n_heads: usize, layers: usize },
    
    // --- Deep Learning ---
    NeuralNetwork(Vec<NNLayer>),
    CNN(Vec<CNNLayer>), RNN(Vec<RNNLayer>),
    GAN { generator: Vec<NNLayer>, discriminator: Vec<NNLayer> },
    VAE { encoder: Vec<NNLayer>, decoder: Vec<NNLayer>, latent_dim: usize },
    DiffusionModel { unet: Vec<NNLayer>, steps: usize },
    
    // --- Reinforcement Learning ---
    DQN { state_dim: usize, action_dim: usize },
    PPO { policy_net: Vec<NNLayer>, value_net: Vec<NNLayer> },
    SAC { actor: Vec<NNLayer>, critic: Vec<NNLayer> },
    A3C { policy_net: Vec<NNLayer>, value_net: Vec<NNLayer> },
    
    // --- Feature Engineering ---
    OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler,
    TfidfVectorizer { max_features: Option<usize> },
    Word2Vec { dim: usize, window: usize },
    FeatureHasher { n_features: usize },
}

#[derive(Debug, Clone)]
pub enum Kernel { Linear, RBF(f64), Polynomial(u32, f64), Sigmoid(f64, f64) }
#[derive(Debug, Clone)]
pub enum GPKernel { SquaredExponential, Matern(f64), RationalQuadratic, Periodic }
#[derive(Debug, Clone)]
pub enum NBKind { Gaussian, Multinomial, Bernoulli }
#[derive(Debug, Clone)]
pub enum ESKind { Simple, Double, Triple }
#[derive(Debug, Clone)]
pub struct XGBoostParams { pub max_depth: u32, pub learning_rate: f64, pub n_estimators: u32, pub objective: String }
#[derive(Debug, Clone)]
pub struct LGBMParams { pub max_depth: i32, pub learning_rate: f64, pub n_estimators: u32, pub num_leaves: u32 }
#[derive(Debug, Clone)]
pub struct CatBoostParams { pub depth: u32, pub learning_rate: f64, pub iterations: u32 }
#[derive(Debug, Clone)]
pub enum NNLayer { Dense(usize, String), Conv2D(usize, usize, String), MaxPool(usize), AvgPool(usize), Dropout(f64), BatchNorm, LayerNorm, Attention { heads: usize, dim: usize }, Embedding(usize, usize), LSTM_(usize), GRU(usize), Residual(Vec<NNLayer>), Flatten, Reshape(Vec<usize>) }
#[derive(Debug, Clone)]
pub enum CNNLayer { Conv2D { filters: usize, kernel: usize, stride: usize, padding: String, activation: String }, MaxPool2D(usize), BatchNorm2D, Dropout2D(f64), GlobalAvgPool2D, Flatten }
#[derive(Debug, Clone)]
pub enum RNNLayer { LSTM { hidden: usize, bidirectional: bool }, GRU { hidden: usize, bidirectional: bool }, Attention { heads: usize } }


// ============================================================================
// SECTION 4: DATA ENGINEERING & SCRAPING
// ============================================================================

/// A data pipeline — ETL/ELT operations as first-class language constructs.
#[derive(Debug, Clone)]
pub struct DataPipeline {
    pub name: String,
    pub sources: Vec<DataSource>,
    pub transforms: Vec<DataTransform>,
    pub sinks: Vec<DataSink>,
    pub schedule: Option<PipelineSchedule>,
    pub error_handling: PipelineErrorHandling,
    pub monitoring: PipelineMonitoring,
}

#[derive(Debug, Clone)]
pub enum DataSource {
    // --- File-based ---
    CSV { path: String, delimiter: char, header: bool, encoding: String },
    Parquet { path: String },
    JSON { path: String, lines: bool },
    Avro { path: String },
    ORC { path: String },
    Excel { path: String, sheet: Option<String> },
    XML { path: String, xpath: String },
    YAML { path: String },
    Arrow { path: String },
    FixedWidth { path: String, widths: Vec<usize> },
    
    // --- Database ---
    SQL { connection: String, query: String },
    PostgreSQL { connection: String, table: String },
    MySQL { connection: String, table: String },
    SQLite { path: String, table: String },
    MongoDB { connection: String, collection: String, filter: String },
    Redis { connection: String, pattern: String },
    Cassandra { connection: String, keyspace: String, table: String },
    DuckDB { path: String, query: String },
    ClickHouse { connection: String, query: String },
    
    // --- Streaming ---
    Kafka { brokers: Vec<String>, topic: String, group: String },
    Pulsar { service_url: String, topic: String },
    RabbitMQ { url: String, queue: String },
    WebSocket { url: String },
    SSE { url: String },
    
    // --- Cloud ---
    S3 { bucket: String, key: String, region: String },
    GCS { bucket: String, object: String },
    Azure { container: String, blob: String },
    BigQuery { project: String, dataset: String, table: String },
    Snowflake { account: String, warehouse: String, query: String },
    Databricks { workspace: String, table: String },
    Redshift { connection: String, query: String },
    
    // --- Web scraping ---
    HTTP { url: String, method: String, headers: HashMap<String, String> },
    REST { base_url: String, endpoint: String, auth: Option<AuthConfig> },
    GraphQL { url: String, query: String, variables: HashMap<String, String> },
    WebScrape { url: String, selector: String, pagination: Option<PaginationConfig> },
    RSS { url: String },
    Sitemap { url: String },
    
    // --- APIs ---
    OpenAPI { spec_url: String, operation: String },
    GRPC { endpoint: String, service: String, method: String },
    
    // --- Real-time ---
    Sensor { device: String, channel: String, sample_rate: f64 },
    MQTT { broker: String, topic: String },
    OPC_UA { endpoint: String, node_id: String },
}

#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub kind: AuthKind,
}

#[derive(Debug, Clone)]
pub enum AuthKind {
    Bearer(String), Basic(String, String), OAuth2 { client_id: String, client_secret: String, token_url: String },
    ApiKey { header: String, key: String }, Certificate { cert_path: String, key_path: String },
}

#[derive(Debug, Clone)]
pub struct PaginationConfig {
    pub strategy: PaginationStrategy,
    pub max_pages: Option<usize>,
    pub delay_ms: u64,
}

#[derive(Debug, Clone)]
pub enum PaginationStrategy {
    NextLink(String),           // CSS selector for "next" link
    PageNumber { param: String, start: usize },
    Cursor { param: String },
    Offset { param: String, limit: usize },
    InfiniteScroll { selector: String },
}

/// Data transformation steps.
#[derive(Debug, Clone)]
pub enum DataTransform {
    Map { expr: String },
    Filter { predicate: String },
    FlatMap { expr: String },
    GroupBy { keys: Vec<String>, aggs: Vec<Aggregation> },
    Join { source: String, on: Vec<(String, String)>, how: JoinKind },
    Sort { by: Vec<String>, ascending: Vec<bool> },
    Deduplicate { subset: Option<Vec<String>> },
    WindowFunction { partition: Vec<String>, order: Vec<String>, func: String },
    Pivot { index: String, columns: String, values: String },
    Unpivot { id_vars: Vec<String>, value_vars: Vec<String> },
    Schema { operations: Vec<SchemaOp> },
    Clean { operations: Vec<CleanOp> },
    Validate { rules: Vec<ValidationRule> },
    Enrich { source: String, key: String },
    Anonymize { columns: Vec<String>, method: AnonymizationMethod },
    Encrypt { columns: Vec<String>, key: String },
    Compress { codec: CompressionCodec },
    Cache { strategy: CacheStrategy },
    Checkpoint { path: String },
    Quality { checks: Vec<DataQualityCheck> },
    Custom { code: String },
}

#[derive(Debug, Clone)]
pub enum SchemaOp { Rename(String, String), Cast(String, DataType), Add(String, DataType, String), Drop(String), Reorder(Vec<String>) }
#[derive(Debug, Clone)]
pub enum CleanOp { TrimWhitespace(String), RemoveNulls(String), FillNulls(String, FillStrategy), RemoveDuplicates, NormalizeText(String), ParseDate(String, String), RegexReplace(String, String, String), Coalesce(Vec<String>, String) }
#[derive(Debug, Clone)]
pub struct ValidationRule { pub column: String, pub rule: ValidationType, pub on_fail: OnValidationFail }
#[derive(Debug, Clone)]
pub enum ValidationType { NotNull, Unique, InRange(f64, f64), Regex(String), InSet(Vec<String>), Custom(String), ForeignKey(String, String) }
#[derive(Debug, Clone)]
pub enum OnValidationFail { Warn, Error, Drop, Quarantine(String), Default(String) }
#[derive(Debug, Clone)]
pub enum AnonymizationMethod { Hash, Mask, Generalize, Suppress, PseudonymizeConsistent }
#[derive(Debug, Clone)]
pub enum CompressionCodec { Snappy, Gzip, LZ4, Zstd, Brotli }
#[derive(Debug, Clone)]
pub enum CacheStrategy { Memory, Disk(String), Redis(String), TTL(Duration) }
#[derive(Debug, Clone)]
pub enum DataQualityCheck { Completeness(f64), Uniqueness(String), Freshness(Duration), Consistency(String), Accuracy(String, f64), Schema(Schema) }

/// Data sinks — where processed data goes.
#[derive(Debug, Clone)]
pub enum DataSink {
    File { path: String, format: OutputFormat },
    Database { connection: String, table: String, mode: WriteMode },
    S3 { bucket: String, key: String, format: OutputFormat },
    Kafka { brokers: Vec<String>, topic: String },
    API { url: String, method: String },
    Dashboard { name: String, plots: Vec<PlotKind> },
    Notification { channel: NotificationChannel, condition: String },
    DataCatalog { catalog: String, dataset: String },
}

#[derive(Debug, Clone)]
pub enum WriteMode { Append, Overwrite, Upsert(Vec<String>), Merge(String) }
#[derive(Debug, Clone)]
pub enum NotificationChannel { Email(String), Slack(String), PagerDuty(String), Webhook(String) }
#[derive(Debug, Clone)]
pub enum PipelineSchedule { Cron(String), Interval(Duration), Event(String), Manual }
#[derive(Debug, Clone)]
pub enum PipelineErrorHandling { FailFast, SkipBad, Retry { max: u32, backoff_ms: u64 }, DeadLetter(String), Circuit { threshold: u32, timeout: Duration } }
#[derive(Debug, Clone)]
pub struct PipelineMonitoring { pub metrics: Vec<String>, pub alerts: Vec<(String, String)>, pub lineage: bool, pub profiling: bool }


// ============================================================================
// SECTION 5: EXTREMELY FINE-GRAINED MANUAL MEMORY MANAGEMENT
// ============================================================================
// Chronos gives you C-level control when you need it, with the safety
// guarantees of linear types unless you explicitly opt out with `unsafe`.

/// Memory layout control — byte-level precision over how data is stored.
#[derive(Debug, Clone)]
pub struct MemoryLayout {
    pub alignment: usize,
    pub padding: PaddingStrategy,
    pub field_order: FieldOrder,
    pub bit_fields: Vec<BitField>,
}

#[derive(Debug, Clone)]
pub enum PaddingStrategy {
    Default,                     // Platform ABI default
    Packed,                      // No padding (#[repr(packed)])
    Aligned(usize),             // Specific alignment
    CacheLine,                  // Align to cache line (usually 64 bytes)
    Page,                       // Align to page boundary (4096 bytes)
    SIMD(usize),                // Align for SIMD (16, 32, 64 bytes)
}

#[derive(Debug, Clone)]
pub enum FieldOrder { Declaration, Size, Alignment, CLayout, Custom(Vec<usize>) }

#[derive(Debug, Clone)]
pub struct BitField { pub name: String, pub bits: u8, pub signed: bool }

/// Custom allocator interface. Chronos programs can define their own
/// allocators and use them for specific types or scopes.
#[derive(Debug, Clone)]
pub enum AllocatorKind {
    /// System allocator (malloc/free).
    System,
    /// Arena / bump allocator — O(1) alloc, bulk free.
    Arena { size: usize },
    /// Pool allocator — fixed-size blocks, O(1) alloc and free.
    Pool { block_size: usize, count: usize },
    /// Slab allocator — multiple pools for different sizes.
    Slab { tiers: Vec<(usize, usize)> },
    /// Stack allocator — LIFO allocation, ultra-fast.
    Stack { size: usize },
    /// Buddy allocator — power-of-two splitting.
    Buddy { min_block: usize, max_block: usize },
    /// TLSF (Two-Level Segregated Fit) — O(1) real-time allocator.
    TLSF { size: usize },
    /// Region-based — tied to a scope, freed when scope exits.
    Region { name: String },
    /// Memory-mapped file backed.
    MMap { path: String, size: usize },
    /// Device memory (GPU/NPU).
    Device { device: String, size: usize },
    /// Huge pages (2MB/1GB pages for reduced TLB pressure).
    HugePage { page_size: HugePageSize },
    /// Shared memory (inter-process communication).
    SharedMemory { name: String, size: usize },
    /// Custom allocator (user-provided alloc/free functions).
    Custom { name: String },
}

#[derive(Debug, Clone)]
pub enum HugePageSize { TwoMB, OneGB }

/// Memory introspection — query the state of the allocator at runtime.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub total_freed: usize,
    pub current_usage: usize,
    pub peak_usage: usize,
    pub allocation_count: u64,
    pub free_count: u64,
    pub fragmentation_ratio: f64,
    pub largest_free_block: usize,
    pub page_faults: u64,
    pub cache_misses: u64,
}

/// Memory operations available to the programmer (in `unsafe` blocks).
/// These are the lowest-level primitives Chronos exposes.
#[derive(Debug, Clone)]
pub enum MemoryOp {
    /// Allocate raw memory from a specific allocator.
    RawAlloc { allocator: AllocatorKind, size: usize, align: usize },
    /// Free raw memory.
    RawFree { ptr: usize, allocator: AllocatorKind },
    /// Reallocate (grow or shrink).
    RawRealloc { ptr: usize, old_size: usize, new_size: usize, align: usize },
    /// Zero-initialize a memory range.
    MemZero { ptr: usize, size: usize },
    /// Copy memory (non-overlapping).
    MemCopy { dst: usize, src: usize, size: usize },
    /// Move memory (overlapping-safe).
    MemMove { dst: usize, src: usize, size: usize },
    /// Set memory to a byte value.
    MemSet { ptr: usize, value: u8, size: usize },
    /// Compare memory regions.
    MemCmp { a: usize, b: usize, size: usize },
    /// Prefetch a cache line.
    Prefetch { ptr: usize, locality: PrefetchLocality },
    /// Memory fence / barrier.
    Fence(MemoryOrdering),
    /// Atomic compare-and-swap.
    CAS { ptr: usize, expected: u64, desired: u64, ordering: MemoryOrdering },
    /// Pin memory (prevent paging out).
    MLock { ptr: usize, size: usize },
    /// Unpin memory.
    MUnlock { ptr: usize, size: usize },
    /// Advise the kernel about memory usage patterns.
    MAdvise { ptr: usize, size: usize, advice: MAdvice },
    /// Get the physical address of a virtual address (requires privileges).
    VirtToPhys { virt_addr: usize },
    /// Map a physical address range (for hardware register access).
    MapPhysical { phys_addr: usize, size: usize },
}

#[derive(Debug, Clone)]
pub enum PrefetchLocality { NonTemporal, Low, Medium, High }
#[derive(Debug, Clone)]
pub enum MemoryOrdering { Relaxed, Acquire, Release, AcqRel, SeqCst }
#[derive(Debug, Clone)]
pub enum MAdvice { Normal, Sequential, Random, WillNeed, DontNeed, HugePage, Free }

/// Placement new — construct an object at a specific memory address.
/// This is the ultimate in manual memory control.
/// Chronos syntax: `let obj = MyStruct::place_at(addr) { field1: val1, ... };`
#[derive(Debug, Clone)]
pub struct PlacementNew {
    pub type_name: String,
    pub address: usize,
    pub fields: Vec<(String, String)>,
}

/// Memory-mapped I/O — for embedded systems and hardware access.
/// Chronos provides type-safe wrappers around MMIO registers.
#[derive(Debug, Clone)]
pub struct MMIORegister {
    pub name: String,
    pub base_address: usize,
    pub size: usize,
    pub fields: Vec<RegisterField>,
    pub access: RegisterAccess,
}

#[derive(Debug, Clone)]
pub struct RegisterField {
    pub name: String,
    pub bit_offset: u8,
    pub bit_width: u8,
    pub access: RegisterAccess,
    pub reset_value: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum RegisterAccess { ReadOnly, WriteOnly, ReadWrite, ReadClear, ReadSet }
