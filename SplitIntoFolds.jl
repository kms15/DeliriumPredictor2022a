#!/usr/bin/julia
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "featurefile"
            help = "file with MRN, DTS and feature columns"
            required = true

        "foldassignmentfile"
            help = "file with MRN, holdout, and (0-based) fold assignments"
            required = true

        "outputprefix"
            help = "prefix for output files; #.csv will be appended (#=fold)"
            required = true

        "--numfolds","-n"
            help = "total number of folds"
            arg_type = Int
            default = 10

        "--useholdout"
            help = "use the holdout set (instead of everything else)"
            action = :store_true
    end

    return parse_args(s)
end

function readfeatureline(stream::IO)
    if eof(stream::IO)
        return typemax(Int),""
    else
        line = readline(stream)
        mrn = parse(Int, split(line, ",", limit=2)[1])
        return mrn, line
    end
end

function main()
    args = parse_commandline()

    featurefile = open(args["featurefile"], "r")
    foldassignmentfile = open(args["foldassignmentfile"], "r")
    outputfiles = [open(args["outputprefix"] * "$i.csv", "w")
                   for i in 0:(args["numfolds"] - 1)]

    # read the headers
    foldassignmentheaders = readline(foldassignmentfile)
    @assert foldassignmentheaders == "MRN,Holdout,CVGroup"
    featureheaders = readline(featurefile)
    for f in outputfiles
        write(f, featureheaders)
        write(f, "\n")
    end

    featuremrn = -1
    featureline = ""
    for patientinfo in eachline(foldassignmentfile)

        # parse the info for this patient
        mrntxt, holdouttxt, cvgrouptxt = split(patientinfo, ",")
        mrn = parse(Int, mrntxt)
        holdout = occursin(r"true"i, holdouttxt)
        cvgroup = parse(Int, cvgrouptxt)

        # read in features until we get to this patient
        while featuremrn < mrn
            featuremrn, featureline = readfeatureline(featurefile)
        end

        # read in the features for this patient
        while featuremrn == mrn
            # if they are not in the holdout group (or the holdout group is
            # being used and they are in the holdout group)
            if holdout == args["useholdout"]
                # copy the line to the appropriate output file
                write(outputfiles[cvgroup+1], featureline)
                write(outputfiles[cvgroup+1], "\n")
            end
            featuremrn, featureline = readfeatureline(featurefile)
        end
    end
end

main()
