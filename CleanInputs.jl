#!/usr/bin/julia
using ArgParse
using OnlineStats

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "sourcefiles"
            help = "files with features, all sharing the same columns"
            nargs = '+'
            required = true

        "--model"
            help = "model file to train or apply"
            required = true

        "--output", "-o"
            help = "target file with cleaned-up features"
            required = false
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

    numfixedcols = 2 # MRN and DTS
    numfeatures = length(allcols[1]) - numfixedcols
    featurenames = allcols[1][numfixedcols+1:end]

    # if we don't have an output, we must be training the model
    if args["output"] == nothing

        # set up the statistics to track

        numnans = [0 for i in 1:numfeatures]
        firstquartiles = [P2Quantile(0.25) for i in 1:numfeatures]
        thirdquartiles = [P2Quantile(0.75) for i in 1:numfeatures]
        medians = [P2Quantile(0.5) for i in 1:numfeatures]
        maxes = [0f0 for i in 1:numfeatures]
        numzeros = [0 for i in 1:numfeatures]
        nonzerothirdquartiles = [P2Quantile(0.75) for i in 1:numfeatures]

        numrows = 0

        # for each source file...
        for f in sourcefiles
            print("processing file $f...\n")
            # read each line
            for line in eachline(f)
                numrows += 1
                if (numrows % 1000) == 0
                    print("Processing row $numrows...\n")
                end
                cols = split(line,",")
                if (length(cols) != numfixedcols + numfeatures)
                    print("error on line $numrows: expected $(numfixedcols + numfeatures) columns but found $(length(cols))\n")
                    exit(1)
                end
                vals = [parse(Float32, val) for val in cols[numfixedcols+1:end]]
                for i in 1:numfeatures
                    # if it's finite, include it in our stats
                    if isfinite(vals[i])
                        fit!(firstquartiles[i], vals[i])
                        fit!(thirdquartiles[i], vals[i])
                        fit!(medians[i], vals[i])
                        maxes[i] = max(maxes[i], vals[i])
                        if vals[i] == 0
                            numzeros[i] += 1
                        else
                            fit!(nonzerothirdquartiles[i], vals[i])
                        end
                    else
                        numnans[i] += 1
                    end
                end
            end
        end

        # write out our stats to the model file
        open(args["model"], "w") do f
            # write the headers
            write(f, "Feature,NaNFraction,Q1,Median,Q3,Max,ZeroFraction,NonZeroQ3\n")
            # write a row of stats for each feature
            for i in 1:numfeatures
                write(f, "$(featurenames[i]),$(numnans[i]/numrows)," *
                      "$(value(firstquartiles[i]))," *
                      "$(value(medians[i]))," *
                      "$(value(thirdquartiles[i]))," *
                      "$(maxes[i])," *
                      "$(numzeros[i]/numrows)," *
                      "$(value(nonzerothirdquartiles[i]))\n"
                     )
            end
        end
    else
        # we must be applying the model to data

        # load the model
        nanfractions = Array{Float32}(undef, 0)
        firstquartiles = Array{Float32}(undef, 0)
        thirdquartiles = Array{Float32}(undef, 0)
        medians = Array{Float32}(undef, 0)
        maxes = Array{Float32}(undef, 0)
        zerofractions = Array{Float32}(undef, 0)
        nonzerothirdquartiles = Array{Float32}(undef, 0)

        open(args["model"], "r") do f
            headers = readline(f)
            @assert headers == "Feature,NaNFraction,Q1,Median,Q3,Max,ZeroFraction,NonZeroQ3"

            for line in eachline(f)
                featurename,nanfraction,q1,median,q3,max_,zerofraction,nonzeroq3 = split(line, ",")
                @assert featurename == featurenames[length(nanfractions)+1]
                push!(nanfractions, parse(Float32, nanfraction))
                push!(firstquartiles, parse(Float32, q1))
                push!(medians, parse(Float32, median))
                push!(thirdquartiles, parse(Float32, q3))
                push!(maxes, parse(Float32, max_))
                push!(zerofractions, parse(Float32, zerofraction))
                push!(nonzerothirdquartiles, parse(Float32, max_))
            end
            @assert length(nanfractions) == length(featurenames)
        end

        # define some limits for removing outliers
        maxnanfraction = 0.95f0
        maxzerofraction = 0.95f0
        minvalid = -6f0 # IQR from median
        maxvalid =  6f0 # IQR from median
        iqrs = Array{Float32}(undef, 0)
        for i in 1:length(nanfractions)
            if zerofractions[i] >= 0.75f0
                # some features (such as 1-hot vectors and medication doses)
                # may be sparse but still significant; we address these by
                # spoofing the IQR to be the third quartile of non-zero values.
                push!(iqrs, nonzerothirdquartiles[i] - firstquartiles[i])
            else
                push!(iqrs, thirdquartiles[i] - firstquartiles[i])
            end
        end
        keepfeatures = (nanfractions .< maxnanfraction) .& (zerofractions .< maxzerofraction)

        open(args["output"], "w") do outfile
            # write the header containing the columns we're keeping
            write(outfile, join(allcols[1][1:numfixedcols], ","), ",",
                  (join(featurenames[keepfeatures],",")), "\n")

            # for each source file...
            numrows = 0
            for f in sourcefiles
                # read each line
                for line in eachline(f)
                    numrows += 1
                    if (numrows % 1000) == 0
                        print("Processing row $numrows...\n")
                    end
                    cols = split(line,",")
                    vals = [parse(Float32, val)
                            for val in cols[numfixedcols+1:end]]

                    # write the fixed columns (e.g. MRN and DTS)
                    write(outfile, join(cols[1:numfixedcols], ","))

                    for i in 1:numfeatures
                        # if we're keeping this column...
                        if keepfeatures[i]
                            # scale the value to have a median of zero
                            # and IQR of 1
                            scaledval = (vals[i] - medians[i]) / iqrs[i]

                            # if it's present and not an outlier, keep the
                            # scaled value, otherwise replace it with 0
                            # (i.e. the median)
                            if minvalid ≤ scaledval ≤ maxvalid
                                write(outfile, ",$scaledval")
                            else
                                write(outfile, ",0")
                            end
                        end
                    end

                    write(outfile, "\n")
                end
            end
        end
    end
end

main()
