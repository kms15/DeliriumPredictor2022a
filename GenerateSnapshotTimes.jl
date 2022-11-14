#!/usr/bin/julia
using ArgParse
using Dates

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "camscreens"
            help = "file with MRN and timestamps for each CAM screen"
            required = true

        "outputfile"
            help = "file to generate with the snapshot MRNs and timestamps"
            required = true
    end

    return parse_args(s)
end

# round backwards to the nearest date time at the given time (e.g. the most recent 3:15 am)
function floorattime(dts::DateTime, time::Time)
    delta = time - Time(0)
    return floor(dts - delta, Day) + delta
end

#@testset "floorattime" begin
#    @test floorattime(DateTime(1992,11,12,3,15), Time(3,15)) == DateTime(1992,11,12,3,15)
#    @test floorattime(DateTime(1992,11,12,3,20), Time(3,15)) == DateTime(1992,11,12,3,15)
#    @test floorattime(DateTime(1992,11,12,3,14), Time(3,15)) == DateTime(1992,11,11,3,15)
#end


function processbatch(batch_mrns, batch_dts, outfile)
    # pre-rounding begins at 5 am
    roundingtime = Time(5,0)

    # round the first time backwards to the start of rounding
    prev_dts = floorattime(batch_dts[1], roundingtime)

    # write the first timestamp
    write(outfile, "$(batch_mrns[1]),$(prev_dts)\n")

    # write any remaining timestamps if they have unique MRNs and times
    for i in 2:length(batch_mrns)
        dts = floorattime(batch_dts[i], roundingtime)
        if dts != prev_dts || batch_mrns[i] != batch_mrns[i-1]
            write(outfile, "$(batch_mrns[i]),$(dts)\n")
            prev_dts = dts
        end
    end
end

function main()
    args = parse_commandline()
    print("generating $(args["outputfile"]) from $(args["camscreens"])\n")

    #camscreens = CSV.File(args["camscreens"], allowmissing=:none,
    #    missingstring="nonesuch", dateformat=dateformat"y-m-d H:M:S",
    #    types=Dict("MRN"=>Int64, "RecordedDTS"=>DateTime))

    open(args["camscreens"], "r") do camscreens
        open(args["outputfile"], "w") do outfile
            write(outfile, "MRN,DTS\n")

            batch_mrns = Array{Int,1}(undef,0)
            batch_dts = Array{DateTime,1}(undef,0)
            minbatchsize = 1000

            headers = split(readline(camscreens), ",")
            mrnindex = findfirst(isequal("MRN"), headers)
            dtsindex = findfirst(isequal("RecordedDTS"), headers)

            for line in eachline(camscreens)
                fields = split(line,",")
                mrn = parse(Int, fields[mrnindex])
                dts = DateTime(fields[dtsindex], dateformat"yyyy-mm-dd HH:MM:SS")

                # if this is a new patient and we have enough prior patients for a
                # batch, then process the prior patients.
                if length(batch_mrns) > minbatchsize && batch_mrns[end] != mrn
                    processbatch(batch_mrns, batch_dts, outfile)
                    empty!(batch_mrns)
                    empty!(batch_dts)
                end

                push!(batch_mrns, mrn)
                push!(batch_dts, dts)
            end

            # process the last batch
            processbatch(batch_mrns, batch_dts, outfile)
        end
    end
end

main()
