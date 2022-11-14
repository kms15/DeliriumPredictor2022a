#!/usr/bin/julia
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "sourcefiles"
            help = "files with features, all sharing the same columns"
            nargs = '+'
            required = true

        "--output", "-o"
            help = "target file with cleaned-up features"
            required = true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    # open the source files and read their headers
    sourcefiles = [open(filename,"r") for filename in args["sourcefiles"]]
    allcols = [split(readline(f),",") for f in sourcefiles]

    # make sure the column names are the same for all input files
    for cols in allcols[2:end]
        @assert all(cols .== allcols[1])
    end

    open(args["output"], "w") do outfile
        # write the header containing the columns we're keeping
        write(outfile, join(allcols[1], ","), "\n")

        # for each source file...
        numrows = 0
        for f in sourcefiles
            # read each line
            for line in eachline(f)
                numrows += 1
                if (numrows % 1000) == 0
                    print("Processing row $numrows...\n")
                end
                write(outfile, line, "\n")
            end
        end
    end
end

main()
