#!/usr/bin/julia
using ArgParse
using Dates

const iso8601 = dateformat"yyyy-mm-ddTHH:MM:SS"
const iso8601date = dateformat"yyyy-mm-dd"

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "patients"
            help = "file with MRN and BirthDTS of each patient"
            required = true

        "snapshottimes"
            help = "file with MRN and times of each snapshot"
            required = true

        "outfile"
            help = "file to generate with ages at each snapshot"
            required = true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    print("generating $(args["outfile"])...\n")
    open(args["patients"], "r") do patientsfile
        open(args["snapshottimes"], "r") do timesfile
            open(args["outfile"], "w") do outfile

                # read the patient header and find the relevant columns
                patientcolnames = split(readline(patientsfile), ",")
                mrncol = convert(Int,
                    findfirst(isequal("MRN"), patientcolnames))
                birthdtscol = convert(Int,
                    findfirst(isequal("BirthDTS"), patientcolnames))
                gendercol = convert(Int,
                    findfirst(isequal("Gender"), patientcolnames))

                # read the snapshots header
                snapshotheader = readline(timesfile)
                @assert snapshotheader == "MRN,DTS"

                # write the output header
                write(outfile, "MRN,DTS,Age,Male\n")

                # start with a dummy patient
                mrnpatient = -1
                birthdts = now()
                male = false

                for line in eachline(timesfile)
                    mrntxt,dtstxt = split(line, ",")
                    mrn = parse(Int, mrntxt)
                    dts = DateTime(dtstxt, iso8601)

                    # advance to the matching patient
                    while mrnpatient < mrn
                        if eof(patientsfile)
                            mrnpatient = typemax(Int)
                        else
                            # unfortunately some lines may have line feeds
                            # and/or commas wrapped in quotes; we need to
                            # handle these carefully
                            pxline = readline(patientsfile)
                            if '"' in pxline
                                # continue reading until all quotes are closed.
                                while count(isequal('"'),pxline) % 2 == 1
                                    pxline *= readline(eventfile)
                                end
                                quotes = split(pxline, "\"")
                                for i in 2:2:length(quotes)
                                    # remove commas, newlines and carriage
                                    # returns
                                    quotes[i] = replace(quotes[i],
                                                        r"[\n\r,]"=>'_')
                                end
                                pxline = join(quotes, "")
                            end
                            cols = split(pxline,",")
                            mrnpatient = parse(Int, cols[mrncol])
                            birthdts = DateTime(cols[birthdtscol], iso8601date)
                            male = cols[gendercol] == "Male"
                        end
                    end

                    # age (in seconds, sticking to mks units)
                    age = Dates.value(dts - birthdts) / 1000 # ms / s
                    write(outfile,line,',',string(age),male ? ",1\n" : ",0\n")
                end
            end
        end
    end
end

main()
