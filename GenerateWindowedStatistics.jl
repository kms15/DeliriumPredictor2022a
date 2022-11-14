#!/usr/bin/julia
using ArgParse
using Dates
using DataStructures
include("AnalysisWindow.jl")
include("TokenNames.jl")

struct Event
    mrn::Int
    dts::DateTime
    id::Int
    value::Float64
end

const iso8601 = dateformat"yyyy-mm-ddTHH:MM:SS"

function readevent(stream::IO, tokennames::TokenNames)
    if eof(stream)
        # create a dummy event with an MRN that is higher than any valid MRN
        return Event(typemax(Int), now(), 1, 0.0)
    else
        mrn,dts,measurename,value = split(readline(stream),",")
        return Event(parse(Int, mrn), DateTime(dts, iso8601),
                     tokennames.nametoid[measurename], parse(Float64, value))
    end
end

function calcwindowedstats(windowstart, windowend, snapshottimesfile,
                           eventsfile, tokennames, outfile, medstats)

    nummeasures = length(tokennames.names)
    windows = [AnalysisWindow() for i = 1:nummeasures]
    eventsinwindow = Deque{Event}()

    # read the headers
    header = readline(snapshottimesfile)
    @assert header == "MRN,DTS"
    header = readline(eventsfile)
    @assert header == "MRN,DTS,MEASURE,VALUE"

    currentmrn = -1
    nextevent = readevent(eventsfile, tokennames)

    # write the header
    statnames = (medstats ? allmedstatnames : allstatnames)(windows[1])
    colnames = [ "$(measure)_$stat" for stat in statnames,
                measure in tokennames.names ][:]
    write(outfile, "MRN,DTS,$(join(colnames, ","))\n")

    # for each snapshot (composed of an mrn and time)
    for line in eachline(snapshottimesfile)
        cols = split(line,",")
        mrn = parse(Int, cols[1])
        dts = DateTime(cols[2], iso8601)

        # if this snapshot is for a new patient...
        if mrn ≠ currentmrn
            currentmrn = mrn

            # start with empty analysis windows
            for window in windows
                empty!(window)
            end

            # advance the event stream to the events for this patient
            empty!(eventsinwindow)
            while nextevent.mrn < currentmrn
                nextevent = readevent(eventsfile, tokennames)
            end
        end

        # consume any new events
        while (nextevent.mrn == currentmrn && # while still on the same patient
               nextevent.dts < dts + windowend) # and we're not past the window
            # add this point to the appropriate window
            push!(windows[nextevent.id], nextevent.dts, nextevent.value)
            push!(eventsinwindow, nextevent)
            nextevent = readevent(eventsfile, tokennames)
        end

        # remove any expired events
        while (!isempty(eventsinwindow) && # while not at the end of the table
               first(eventsinwindow).dts < dts + windowstart) # & out of window
            # remove this point from the appropriate window
            event = popfirst!(eventsinwindow)
            pop!(windows[event.id], event.dts, event.value)
        end

        # fill in the row for the given snapshot
        results = [measure for imeasure = 1:nummeasures
                   for measure in
                   (medstats ? allmedstats : allstats)(windows[imeasure])]
        write(outfile, "$mrn,$dts,$(join(results,","))\n")
    end
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "windowstart"
            help = "offset of the beginning of the window (hours)"
            arg_type = Int
            required = true

        "windowend"
            help = "offset of the end of the window (hours)"
            arg_type = Int
            required = true

        "windowtimes"
            help = "file with MRN and times of each snapshot"
            required = true

        "tokenizedevents"
            help = "file with MRNs, datetime stamps, measure, and values"
            required = true

        "tokennamesfile"
            help = "file with names of all possible measures"
            required = true

        "outfile"
            help = "file to generate with windowed statistics"
            required = true

        "--medstats"
            help = "use logic for medications (e.g. assume missing data is 0)"
            action = :store_true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    tokennames = readtokennames(args["tokennamesfile"])

    print("generating $(args["outfile"])...\n")
    open(args["windowtimes"], "r") do windowtimesfile
        open(args["tokenizedevents"], "r") do eventsfile
            open(args["outfile"], "w") do outfile
                calcwindowedstats(
                    Hour(args["windowstart"]), Hour(args["windowend"]),
                    windowtimesfile, eventsfile, tokennames, outfile,
                    args["medstats"])
            end
        end
    end
end

main()
