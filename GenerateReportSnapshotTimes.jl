#!/usr/bin/julia
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "patients"
            help = "file with MRNs of current inpatients"
            required = true

        "dts"
            help = "date-time stamp of the report to generate"
            required = true
        "outfile"
            help = "file to generate with ages at each snapshot"
            required = true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    # pre-rounding begins at 5 am
    dts = args["dts"]

    print("generating $(args["outfile"])...\n")
    open(args["patients"], "r") do patientsfile
        open(args["outfile"], "w") do outfile

            # read the patient header and find the relevant columns
            patientcolnames = split(readline(patientsfile), ",")
            mrncol = convert(Int,
                findfirst(isequal("MRN"), patientcolnames))

            # write the output header
            write(outfile, "MRN,DTS\n")

            for line in eachline(patientsfile)
                if '"' in line
                    # continue reading until all quotes are closed.
                    while count(isequal('"'),line) % 2 == 1
                        line *= readline(eventfile)
                    end
                    quotes = split(line, "\"")
                    for i in 2:2:length(quotes)
                        # remove commas, newlines and carriage
                        # returns
                        quotes[i] = replace(quotes[i],
                                            r"[\n\r,]"=>'_')
                    end
                    line = join(quotes, "")
                end
                cols = split(line,",")
                mrn = cols[mrncol]
                write(outfile,mrn,',',dts,'\n')
            end
        end
    end
end

main()
