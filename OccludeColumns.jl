#!/usr/bin/julia

function main()
    if length(ARGS) == 2
        (xfilename,outfilename) = ARGS
    else
        print(stderr, "Usage:\nOccludeColumns.jl INFILE OUTFILE\n")
        return 1
    end

    xfile = open(xfilename, "r")
    outfile = open(outfilename, "w")

    # copy the header
    header = readline(xfile)
    @assert startswith(header, "MRN,DTS")
    write(outfile, header, '\n')

    # read and parse the first line
    line = readline(xfile)
    @assert eof(xfile)
    write(outfile, line, '\n')
    columns = split(line, ',')
    firstfeaturecol = 3

    for i in firstfeaturecol:length(columns)
        write(outfile, columns[1])
        for j in 2:(i-1)
            write(outfile, ',', columns[j])
        end
        write(outfile, ",0")
        for j in (i+1):length(columns)
            write(outfile, ',', columns[j])
        end
        write(outfile, '\n')
    end

    return 0
end

main()
