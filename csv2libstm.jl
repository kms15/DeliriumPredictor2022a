#!/usr/bin/julia

function main()
    if length(ARGS) == 2
        (xfilename,outfilename) = ARGS
        yfilename = nothing
    elseif length(ARGS) == 3
        (xfilename,yfilename,outfilename) = ARGS
    else
        print(stderr, "Usage:\ncsv2svmlib XCSV [YCSV] LIBSVM")
        return 1
    end

    xfile = open(xfilename, "r")
    yfile = (yfilename == nothing) ? nothing : open(yfilename, "y")

    # read the headers
    header = split(readline(xfile),",")
    @assert header[1] == "MRN"
    @assert header[2] == "DTS"
    if yfile != nothing
        header = split(readline(yfile),",")
        @assert header[1] == "MRN"
        @assert header[2] == "DTS"
    end

    outfile = open(outfilename, "w")
    linenum = 2
    while !eof(xfile)
        linenum += 1
        if (linenum % 1000) == 0
            print("processing line $linenum...\n")
        end

        # if we don't have a file with y values...
        if yfile == nothing
            # start the line with a dummy label/y value
            write(outfile, "0")
        else
            # copy the y value as the label for this line
            yval = split(readline(xfile), ',')[3]
            if yval == "True"
                write(outfile, "1")
            elseif yval == "False"
                write(outfile, "0")
            else
                error("unrecognized y value $yval")
            end
        end

        # extract the features from the file with x values
        features = map(x->parse(Float32, x), split(readline(xfile), ',')[3:end])

        # write the non-zero features with their (0 based) index
        for (i,feature) in enumerate(features)
            if feature != 0
                write(outfile, " $(i-1):$feature")
            end
        end
        write(outfile, '\n')
    end
    return 0
end

main()
