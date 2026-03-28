using QuantumClusterTheories
using Documenter

DocMeta.setdocmeta!(QuantumClusterTheories, :DocTestSetup, :(using QuantumClusterTheories); recursive=true)

makedocs(;
    modules=[QuantumClusterTheories],
    authors="waltergu <waltergu1989@gmail.com> and contributors",
    repo="https://github.com/Quantum-Many-Body/QuantumClusterTheories.jl/blob/{commit}{path}#{line}",
    sitename="QuantumClusterTheories.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Quantum-Many-Body.github.io/QuantumClusterTheories.jl",
        assets=["assets/favicon.ico"],
    ),
    pages=[
        "Home" => "index.md",
            "Manual" => [
            "manual/QuantumClusterTheories.md",
        ],
            "Examples" => [
            "examples/Introduction.md",
            "examples/SquareHubbardSpectral.md",
        ]
    ],
)

deploydocs(;
    repo="github.com/Quantum-Many-Body/QuantumClusterTheories.jl",
    devbranch="main",
)
