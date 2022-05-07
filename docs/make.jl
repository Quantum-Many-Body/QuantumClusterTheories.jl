using QuantumClusterTheory
using Documenter

DocMeta.setdocmeta!(QuantumClusterTheory, :DocTestSetup, :(using QuantumClusterTheory); recursive=true)

makedocs(;
    modules=[QuantumClusterTheory],
    authors="waltergu <waltergu1989@gmail.com> and contributors",
    repo="https://github.com/Quantum-Many-Body/QuantumClusterTheory.jl/blob/{commit}{path}#{line}",
    sitename="QuantumClusterTheory.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Quantum-Many-Body.github.io/QuantumClusterTheory.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Quantum-Many-Body/QuantumClusterTheory.jl",
    devbranch="main",
)
