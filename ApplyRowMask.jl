#!/usr/bin/julia
using ArgParse
using OnlineStats

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "infile"
            help = "file with features to filter"
            required = true

        "rowmask"
            help = "row mask file to apply"
            required = true

        "outfile"
            help = "output filename for feature file with rows removed"
            required = true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    # open the source file and read the headers
    infile = open(args["infile"],"r")
    firstline = readline(infile)

    # open the row mask file and read and verify the header
    rowmaskfile = open(args["rowmask"],"r")
    if (readline(rowmaskfile) != "keep")
        error("unexpected rowmask file format")
    end

    # open the destination file and copy the headers
    outfile = open(args["outfile"], "w")
    write(outfile, firstline, '\n')

    # for each line
    numrows = 0
    for (mask,line) in zip(eachline(rowmaskfile),eachline(infile))
        numrows += 1
        if (numrows % 1000) == 0
            print("Processing row $numrows...\n")
        end

        # copy only if it does not match the exclusion criteria
        if mask == "1"
            write(outfile, line, '\n')
        end
    end
end

main()
