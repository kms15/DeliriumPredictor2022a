#!/usr/bin/julia
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "sourcefiles"
            help = "files with features, all sharing the same MRN and DTS columns"
            nargs = '+'
            required = true

        "--output","-o"
            help = "file to generate with the merged features"
            required = true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    sourcefiles = [open(filename,"r") for filename in args["sourcefiles"]]
    outfile = open(args["output"], "w")

    # read the headers
    headersets = [split(readline(f),",") for f in sourcefiles]

    # write the merged headers
    write(outfile, "MRN,DTS")
    for headerset in headersets
        @assert headerset[1] == "MRN"
        @assert headerset[2] == "DTS"
        write(outfile, "," * join(headerset[3:end], ","))
    end
    write(outfile, "\n")

    linenum = 1
    while !eof(sourcefiles[1])
        linenum += 1
        if (linenum % 1000) == 0
            print("processing line $linenum...\n")
        end
        lines = [readline(f) for f in sourcefiles]
        write(outfile, lines[1])
        mrndts_slice = findfirst(r"^[^,]*,[^,]*,", lines[1])
        datastart = last(mrndts_slice)
        for line in lines[2:end]
            @assert line[mrndts_slice] == lines[1][mrndts_slice]
            write(outfile, line[datastart:end])
        end
        write(outfile, '\n')
    end
end

main()
