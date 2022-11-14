#!/usr/bin/julia
using ArgParse
using OnlineStats

# extracts the nth column from a comma-delimited line of text
function extractcol(colnum, line)
    step = 1 + 1 # number of character to advance from end of col
                 # to start of next col
    slice = 0:-step
    for i in 1:colnum
        slice = findnext(r"[^,]*", line, last(slice) + step)
    end
    return parse(Float32, line[slice])
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "inputfile"
            help = "file with features to filter"
            required = true

        "maskfile"
            help = "target filename for the row mask file"
            required = true

        "--column"
            help = "column to filter on"
            required = true

        "--removeifgreaterthan"
            help = "exclude a row if the filtered column is greater than this value"
            required = true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    # open the source file and read the headers
    inputfile = open(args["inputfile"],"r")
    firstline = readline(inputfile)
    allcols = split(firstline,",")
    filtercol = convert(Int, findfirst(isequal(args["column"]), allcols))
    excludeval = parse(Float32, string(args["removeifgreaterthan"]))

    # open the row mask mask file and write the header
    outputfile = open(args["maskfile"], "w")
    write(outputfile, "keep\n")

    # for each line
    numrows = 0
    for line in eachline(inputfile)
        numrows += 1
        if (numrows % 1000) == 0
            print("Processing row $numrows...\n")
        end

        # keep only if it does not match the exclusion criteria
        if extractcol(filtercol, line) > excludeval
            write(outputfile, "0\n")
        else
            write(outputfile, "1\n")
        end
    end
end

main()
