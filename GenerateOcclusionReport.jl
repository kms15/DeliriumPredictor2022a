#!/usr/bin/julia
using Printf

function main()
    if length(ARGS) == 4
        (xfilename,yfilename,intervenablesfilename,outfilename) = ARGS
    else
        print(stderr, "Usage:\nGenerateOcclusionReport.jl XFILE YFILE INTERVENIABLESFILE REPORTFILE\n")
        return 1
    end


    # read the feature names and values
    featurenames = ["none"]
    featurevals = [0f0]
    mrn = "*"
    open(xfilename, "r") do xfile
        header = readline(xfile)
        @assert startswith(header, "MRN,DTS")
        columns = split(header, ',')
        featurenames = columns[3:end]
        featurevals = map(x->parse(Float32, x), split(readline(xfile), ',')[3:end])
    end

    # read the risk scores
    baserisk = 0
    risks = []
    open(yfilename, "r") do yfile
        header = readline(yfile)
        @assert header == "MRN,DTS,y_hat"
        (mrn,dts,baseriskstr) = split(readline(yfile), ',')
        baserisk = parse(Float32, baseriskstr)
        for l in eachline(yfile)
            push!(risks, parse(Float32, split(l, ',')[3]))
        end
    end

    # read the intervenables file
    intervenables = Set{String}()
    open(intervenablesfilename, "r") do intervenablesfile
        header = readline(intervenablesfile)
        @assert header == "feature"
        for feature in eachline(intervenablesfile)
            push!(intervenables, feature)
        end
    end

    @assert length(featurenames) == length(risks)

    # generate the report
    open(outfilename, "w") do outfile
        write(outfile, "# MRN: $mrn\n")
        write(outfile, "Risk of delirium: $(@sprintf("%.2f", 100*baserisk))%\n")
        write(outfile, "\n")
        write(outfile, "Potentially intervenable contributing factors:\n")
        for i in sortperm(risks)
            # stop the list at the point where occlusions no longer reduce the risk
            if (baserisk - risks[i] <= 0)
                break
            end
            featurename = replace(featurenames[i], "_"=>" ")
            if (in(featurename, intervenables))
                write(outfile, "- $(@sprintf("%5.2f", 100 * (baserisk - risks[i])))%: " *
                      "$(featurename) is " *
                      (featurevals[i] > 0 ? "higher" : "lower") *
                      " than average ($(featurevals[i]) IQR)\n")
            end
        end
        write(outfile, "\nOther contributing factors:\n")
        for i in sortperm(risks)
            # stop the list at the point where occlusions no longer reduce the risk
            if (baserisk - risks[i] <= 0)
                break
            end
            featurename = replace(featurenames[i], "_"=>" ")
            if (!in(featurename, intervenables))
                write(outfile, "- $(@sprintf("%5.2f", 100 * (baserisk - risks[i])))%: " *
                      "$(featurename) is " *
                      (featurevals[i] > 0 ? "higher" : "lower") *
                      " than average ($(featurevals[i]) IQR)\n")
            end
        end
        write(outfile, "\n\n")
    end

    return 0
end

main()
