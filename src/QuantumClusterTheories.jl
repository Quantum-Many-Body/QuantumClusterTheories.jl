module QuantumClusterTheories

include("Optimizer.jl")
using .Optimizer
export NoisyNewton

using LinearAlgebra: I, det, dot, tr
using TimerOutputs: TimerOutput
using Optim: LBFGS, Options, optimize
using QuadGK: quadgk
using QuantumLattices: AbstractLattice, Action, Algorithm, Assignment, Bond, BrillouinZone, CoordinatedIndex, Data, Fock, Frontend, Generator, Hilbert, Index, Metric, Neighbors, OneAtLeast, OneOrMore, ReciprocalSpace, Table, Term
using QuantumLattices: atol, bonds, expand, isannihilation, isintracell, issubordinate, matrix, nneighbor, rank, rcoordinate, reciprocals, rtol
using StaticArrays: SVector
using TightBindingApproximation: Quadraticization, TBA, TBAKind, commutator
import QuantumLattices: Parameters, kind, options, run!, update!
import TightBindingApproximation.Fitting: optimize!

include("Core.jl")
export CPT, CPTPerturbation, DynamicalSpectra, DynamicalSpectraData, ImpuritySolver, Periodization, Perturbation, QCT, VCA, VCAPerturbation, expectation, operators, optimize!, quadratic, qcttimer, Ω

end
