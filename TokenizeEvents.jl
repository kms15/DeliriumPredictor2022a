#!/usr/bin/julia
using ArgParse
include("TokenNames.jl")

function coercetonumber(value)
    m = match(r"[0-9]+(\.[0-9]+)?", value)

    if m != nothing
        return parse(Float64, m.match)
    elseif match(r"negative|Negative|NEGATIVE", value) != nothing
        return 0.
    elseif match(r"positive|Positive|POSITIVE", value) != nothing
        return 1.
    else
        return NaN
    end
end;


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "eventfile"
            help = "file with MRNs, datetime stamps, measure, and values"
            required = true

        "mrncolumn"
            help = "columnname with the patient's MRN"
            required = true

        "dtscolumn"
            help = "columnname with the date-time stamp"
            required = true

        "measurecolumn"
            help = "columnname with the name of the measure"
            required = true

        "valuecolumn"
            help = "columnname with the value of the measure"
            required = true

        "tokeneventsfile"
            help = "file to generate with the tokenized events"
            required = true

        "tokennamesfile"
            help = "file to generate with the tokenized names"
            required = true

        "--synonymsfile"
            help = "file with synonyms used to normalize the names"
    end

    return parse_args(s)
end


function main()
    args = parse_commandline()

    tokens = TokenNames()
    synonyms = Dict{String,String}()

    if args["synonymsfile"] != nothing
        print("reading synonyms...\n")

        open(args["synonymsfile"], "r") do f
            headers = split(readline(f), ",")
            for line in eachline(f)
                name,synonym = split(uppercase(line),",")
                synonyms[synonym] = name
            end
        end
    end

    print("generating $(args["tokeneventsfile"])...\n")
    open(args["eventfile"], "r") do eventfile
        open(args["tokeneventsfile"], "w") do tokeneventsfile
            write(tokeneventsfile, "MRN,DTS,MEASURE,VALUE\n")

            headers = split(readline(eventfile), ",")
            mrnindex = findfirst(isequal(args["mrncolumn"]), headers)
            dtsindex = findfirst(isequal(args["dtscolumn"]), headers)
            measureindex = findfirst(isequal(args["measurecolumn"]), headers)
            valueindex = findfirst(isequal(args["valuecolumn"]), headers)

            for line in eachline(eventfile)
                # unfortunately some lines may have line feeds and/or commas
                # wrapped in quotes; we need to handle these carefully
                if '"' in line
                    # continue reading until all quotes are closed.
                    while count(isequal('"'),line) % 2 == 1
                        line *= readline(eventfile)
                    end
                    quotes = split(line, "\"")
                    for i in 2:2:length(quotes)
                        # remove commas, newlines and carriage returns
                        quotes[i] = replace(quotes[i], r"[\n\r,]"=>'_')
                    end
                    line = join(quotes, "")
                end

                fields = split(line,",")
                mrn = fields[mrnindex]
                dts = replace(fields[dtsindex], " " => "T")
                measurename = uppercase(fields[measureindex])
                measurename = get(synonyms, measurename, measurename)
                valuetxt = fields[valueindex]

                # some values (e.g. BP) have more than one value separated by
                # slashes.
                measurevals = map(coercetonumber, split(valuetxt, "/"))
                for (i,val) in enumerate(measurevals)
                    if isfinite(val)
                        submeasurename = measurename*"_SLASH"^(i-1)
                        submeasurename = get(synonyms, submeasurename,
                            submeasurename)
                        addtoken!(tokens, submeasurename)
                        write(tokeneventsfile,
                            "$mrn,$dts,$submeasurename,$val\n"
                        )
                    end
                end
            end
        end
    end

    open(args["tokennamesfile"], "w") do tokennamesfile
        write(tokennamesfile, "MEASURE\n")

        for name in tokens.names
            write(tokennamesfile, "$name\n")
        end
    end
end

main()
