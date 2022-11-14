mutable struct TokenNames
     names::Array{String,1}
     counts::Array{Int,1}
     nametoid::Dict{String,Int}
     TokenNames() = new([], [], Dict())
end

function addtoken!(tokens, name)
    id = (get!(tokens.nametoid, name) do
        push!(tokens.names, name)
        push!(tokens.counts, 0)
        length(tokens.names)
    end)

    tokens.counts[id] += 1
    return id
end

function readtokennames(filename::String)
    result = TokenNames()

    open(filename, "r") do f
        header = readline(f)
        @assert header == "MEASURE"

        for line in eachline(f)
            addtoken!(result, line)
        end
    end

    return result
end
