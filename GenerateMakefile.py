#!/usr/bin/python3
import os
import numpy as np

nfolds = 10
dpi = 1200
num_bootstraps=1000
datadate = "2019-01-17"
default_reportdts = "2018-01-17T05:00:00"
reportdts = "$(REPORTDTS)"
reportsuffix = "$(REPORTSUFFIX)"
numpatientreports = 20
importeddatadir = f"ImportedData/{datadate}/"
cachedir = f"Cache/{datadate}/"
outputsdir = f"Outputs/{datadate}/"
xtimes = [(-24,0)] #, (-48,-24)]
ytimes = [(0, 24)] #, (0, 48), (0, 720)]
modelnames = ['Boosted Trees', 'Random Forest', 'Multilayer Perceptron',
        'Logistic Regression',
        #'Support Vector Classifier'
        ]
modelprefixes = ['bt', 'rf', 'mlp', 'lr',
        #'svc'
        ]
modelscripts = ['./BoostedTrees.py', './RandomForest.py',
        './MLP.py', './LogisticRegression.py',
        #'./NuSVC.py'
        ]
modelhasimportance = [ True, True, False, True, False ]
imported_measure_tables = [
    # BaseName      # DTS column            # Measure name column   # Value column      # use medstats
    [ "Lab",        "ResultDTS",            "ComponentNM",          "Result",           False   ],
    [ "Vital",      "RecordedDTS",          "FlowsheetMeasureNM",   "MeasureTXT",       False   ],
    [ "CamScore",   "RecordedDTS",          "FlowsheetMeasureNM",   "MeasureTXT",       False   ],
    [ "Medication", "MedicationTakenDTS",   "MedicationDSC",        "DiscreteDoseAMT",  True    ],
]

conditions = ['', 'notdelirious_', 'nopriordelirium_']
conditionnames = ['All Patients', 'Not Delirious', 'No Prior Delirium']

# make the outputs directory (if it does not yet exist)
try:
    os.mkdir(outputsdir)
except OSError:
    pass # directory already exists

def start_rule(f, targets, dependencies):
    if len(targets) == 1:
        f.write(f"{targets[0]} : {' '.join(dependencies)}\n")
    else:
        # link the multiple targets to a single build rule using a dummy
        # intermediate file
        intermediate = f"intermediate{hash(' '.join(targets))}"
        f.write(f"{' '.join(targets)} : {intermediate}\n")
        f.write(f"\t@\n\n")
        f.write(f".INTERMEDIATE : {intermediate}\n")
        f.write(f"{intermediate} : {' '.join(dependencies)}\n")

with open("Makefile", "w") as f:
    plots = []
    text_results = []

    # add variable definitions
    f.write(f"REPORTDTS={default_reportdts}\n\n")
    f.write(f"REPORTSUFFIX=_$(subst :,,$(REPORTDTS))\n\n")

    # if no specific make target is given, generate all plots
    f.write(f"all : all_plots all_text_results\n\n")

    # initialized the python virtual environment for shap
    f.write("shap-venv/bin/activate: shap-requirements.txt\n"
            + "\trm -rf shap-venv\n"
            + "\tpython3 -m venv shap-venv\n"
            + "\tshap-venv/bin/pip3 install pip==22.2.2\n"
            + "\tshap-venv/bin/pip3 install -r $<\n"
            + "\n")

    # generate timestamp times for training
    input_files = [
            f"{importeddatadir}CamScores.csv",
            ]
    script = "./GenerateSnapshotTimes.jl"
    targets = [f"{cachedir}SnapshotTimes.csv"]
    dependencies = [script, *input_files]
    start_rule(f, targets, dependencies)
    f.write(f"\t{script} {' '.join(input_files)} {' '.join(targets)}\n")
    f.write("\n")

    # generate timestamp times for a specific report time
    input_files = [
            f"{importeddatadir}Patients.csv",
            ]
    script = "./GenerateReportSnapshotTimes.jl"
    targets = [f"{cachedir}SnapshotTimes{reportsuffix}.csv"]
    dependencies = [script, *input_files]
    start_rule(f, targets, dependencies)
    f.write(f"\t{script} {' '.join(input_files)} {reportdts} {' '.join(targets)}\n")
    f.write("\n")

    # generate Demographics for training
    input_files = [
            f"{importeddatadir}Patients.csv",
            f"{cachedir}SnapshotTimes.csv"
            ]
    script = "./GenerateDemographics.jl"
    targets = [f"{cachedir}Demographics.csv"]
    dependencies = [script, *input_files]
    start_rule(f, targets, dependencies)
    f.write(f"\t{script} {' '.join(input_files)} {' '.join(targets)}\n")
    f.write("\n")

    # generate Demographics for a specific report time
    input_files = [
            f"{importeddatadir}Patients.csv",
            f"{cachedir}SnapshotTimes{reportsuffix}.csv"
            ]
    script = "./GenerateDemographics.jl"
    targets = [f"{cachedir}Demographics{reportsuffix}.csv"]
    dependencies = [script, *input_files]
    start_rule(f, targets, dependencies)
    f.write(f"\t{script} {' '.join(input_files)} {' '.join(targets)}\n")
    f.write("\n")

    # generate CV and holdout groups
    input_files = [
            f"{importeddatadir}Patients.csv",
            ]
    script = "./GenerateHoldoutAndFoldAssignment.py"
    targets = [f"{cachedir}holdout_and_cvgroups.csv",
            f"{cachedir}holdout_and_cvgroups.parquet.snappy"]
    dependencies = [script, *input_files]
    start_rule(f, targets, dependencies)
    f.write(f"\t{script} {' '.join(input_files)} {' '.join(targets)}\n")
    f.write("\n")

    # generate the tokenized versions of the imported data
    for basename,dtscol,measurecol,valuecol,medstats in imported_measure_tables:
        input_files = [
                f"./synonyms.csv",
                f"{importeddatadir}{basename}s.csv",
                ]
        script = "./TokenizeEvents.jl"
        targets = [
                f"{cachedir}{basename}s_tokenized.csv",
                f"{cachedir}{basename}s_tokennames.csv",
                ]
        dependencies = [script, *input_files]
        start_rule(f, targets, dependencies)
        f.write(f"\t{script} --synonyms {' '.join(input_files)} MRN {dtscol} "
            + f"{measurecol} {valuecol} {' '.join(targets)}\n")
        f.write("\n")

    # generate statistics files for training
    for basename,dtscol,measurecol,valuecol,medstats in imported_measure_tables:
        for xstart,xstop in xtimes + ytimes:
            input_files = [
                f"{cachedir}SnapshotTimes.csv",
                f"{cachedir}{basename}s_tokenized.csv",
                f"{cachedir}{basename}s_tokennames.csv",
                ]
            script = "./GenerateWindowedStatistics.jl"
            targets = [
                f"{cachedir}{basename}Statistics_" +
                f"{xstart}hours_to_{xstop}hours.csv",
                ]
            dependencies = [script, *input_files]
            start_rule(f, targets, dependencies)
            f.write(f"\t{script}" +
                (" --medstats " if medstats else " ") +
                f"{xstart} {xstop} {' '.join(input_files)} " +
                f"{' '.join(targets)}\n")
            f.write("\n")

    # generate statistics files for a specific date for reporting
    for basename,dtscol,measurecol,valuecol,medstats in imported_measure_tables:
        for xstart,xstop in xtimes + ytimes:
            input_files = [
                f"{cachedir}SnapshotTimes{reportsuffix}.csv",
                f"{cachedir}{basename}s_tokenized.csv",
                f"{cachedir}{basename}s_tokennames.csv",
                ]
            script = "./GenerateWindowedStatistics.jl"
            targets = [
                f"{cachedir}{basename}Statistics_" +
                f"{xstart}hours_to_{xstop}hours{reportsuffix}.csv",
                ]
            dependencies = [script, *input_files]
            start_rule(f, targets, dependencies)
            f.write(f"\t{script}" +
                (" --medstats " if medstats else " ") +
                f"{xstart} {xstop} {' '.join(input_files)} " +
                f"{' '.join(targets)}\n")
            f.write("\n")

    # generate X files for training
    for xstart,xstop in xtimes:
        xprefix = f"{cachedir}X{xstart}_{xstop}_"
        input_files = [
                f"{cachedir}LabStatistics_{xstart}hours_to_{xstop}hours.csv",
                f"{cachedir}VitalStatistics_{xstart}hours_to_{xstop}hours.csv",
                f"{cachedir}CamScoreStatistics_{xstart}hours_to_{xstop}hours.csv",
                f"{cachedir}MedicationStatistics_{xstart}hours_to_{xstop}hours.csv",
                f"{cachedir}Demographics.csv",
                ]
        script = "./HCatFeatures.jl"
        targets = [f"{xprefix}X.csv"]
        dependencies = [script, *input_files]
        start_rule(f, targets, dependencies)
        f.write(f"\t{script} {' '.join(input_files)} -o {' '.join(targets)}\n")
        f.write("\n")

    # generate X files for a specific date and time for reporting
    for xstart,xstop in xtimes:
        xprefix = f"{cachedir}X{xstart}_{xstop}_"
        input_files = [
                f"{cachedir}LabStatistics_{xstart}hours_to_{xstop}hours{reportsuffix}.csv",
                f"{cachedir}VitalStatistics_{xstart}hours_to_{xstop}hours{reportsuffix}.csv",
                f"{cachedir}CamScoreStatistics_{xstart}hours_to_{xstop}hours{reportsuffix}.csv",
                f"{cachedir}MedicationStatistics_{xstart}hours_to_{xstop}hours{reportsuffix}.csv",
                f"{cachedir}Demographics{reportsuffix}.csv",
                ]
        script = "./HCatFeatures.jl"
        targets = [f"{xprefix}X{reportsuffix}.csv"]
        dependencies = [script, *input_files]
        start_rule(f, targets, dependencies)
        f.write(f"\t{script} {' '.join(input_files)} -o {' '.join(targets)}\n")
        f.write("\n")

    # generate X folds
    for xstart,xstop in xtimes:
        xprefix = f"{cachedir}X{xstart}_{xstop}_"
        input_files = [ f"{xprefix}X.csv", f"{cachedir}holdout_and_cvgroups.csv" ]
        script = "./SplitIntoFolds.jl"
        targets = [f"{xprefix}X_fold{i}.csv" for i in range(nfolds)]
        dependencies = [script, *input_files]
        start_rule(f, targets, dependencies)
        f.write(f"\t{script} {' '.join(input_files)} {xprefix}X_fold\n")
        f.write("\n")

    # generate y files
    for ystart,ystop in ytimes:
        yprefix = f"{cachedir}y{ystart}_{ystop}_"
        input_files = [
                f"{cachedir}CamScoreStatistics_{ystart}hours_to_{ystop}hours.csv",
                f"{cachedir}holdout_and_cvgroups.parquet.snappy"
                ]
        script = "./GenerateYs.py"
        targets = [f"{yprefix}y_fold{i}.csv" for i in range(nfolds)]
        dependencies = [script, *input_files]
        start_rule(f, targets, dependencies)
        f.write(f"\t{script} {' '.join(input_files)} {yprefix}\n")
        f.write("\n")

    # generate X folds for holdout set
    for xstart,xstop in xtimes:
        xprefix = f"{cachedir}X{xstart}_{xstop}_"
        input_files = [ f"{xprefix}X.csv", f"{cachedir}holdout_and_cvgroups.csv" ]
        script = "./SplitIntoFolds.jl"
        targets = [f"{xprefix}holdout_X_fold{i}.csv" for i in range(nfolds)]
        dependencies = [script, *input_files]
        start_rule(f, targets, dependencies)
        f.write(f"\t{script} --useholdout {' '.join(input_files)} {xprefix}holdout_X_fold\n")
        f.write("\n")

    # generate y files for holdout set
    for ystart,ystop in ytimes:
        yprefix = f"{cachedir}y{ystart}_{ystop}_holdout_"
        input_files = [
                f"{cachedir}CamScoreStatistics_{ystart}hours_to_{ystop}hours.csv",
                f"{cachedir}holdout_and_cvgroups.parquet.snappy"
                ]
        script = "./GenerateYs.py"
        targets = [f"{yprefix}y_fold{i}.csv" for i in range(nfolds)]
        dependencies = [script, *input_files]
        start_rule(f, targets, dependencies)
        f.write(f"\t{script} --useholdout {' '.join(input_files)} {yprefix}\n")
        f.write("\n")

    # generate merged folds
    for holdout in ['', '_holdout']:
        # generate a single X file with all of the folds
        for xstart,xstop in xtimes:
            xprefix = f"{cachedir}X{xstart}_{xstop}"
            input_files = [f"{xprefix}{holdout}_X_fold{j}.csv" for j in range(nfolds)]
            script = "./VCatCSV.jl"
            target = f"{xprefix}{holdout}_X_allfolds.csv"
            dependencies = [script, *input_files]
            f.write(f"{target} : {' '.join(dependencies)}\n")
            f.write(f"\t{script} {' '.join(input_files)} -o {target}\n")
            f.write("\n")

        # generate a single y file with all of the folds
        for ystart,ystop in ytimes:
            yprefix = f"{cachedir}y{ystart}_{ystop}"
            input_files = [f"{yprefix}{holdout}_y_fold{j}.csv" for j in range(nfolds)]
            script = "./VCatCSV.jl"
            target = f"{yprefix}{holdout}_y_allfolds.csv"
            dependencies = [script, *input_files]
            f.write(f"{target} : {' '.join(dependencies)}\n")
            f.write(f"\t{script} {' '.join(input_files)} -o {target}\n")
            f.write("\n")

    # generate counts of patients and days used for the dataset
    for ystart,ystop in ytimes:
        yprefix = f"{cachedir}y{ystart}_{ystop}"
        input_files = [f"{yprefix}_y_allfolds.csv",f"{yprefix}_holdout_y_allfolds.csv"]
        script = "./GeneratePatientCounts.py"
        target = f"{outputsdir}y{ystart}_{ystop}_counts.csv"
        dependencies = [script, *input_files]
        f.write(f"{target} : {' '.join(dependencies)}\n")
        f.write(f"\t{script} {' '.join(input_files)} {target}\n")
        f.write("\n")
        text_results.append(target)

    # Clean the data (impute missing data and drop outliers)
    for xstart,xstop in xtimes:
        xprefix = f"{cachedir}X{xstart}_{xstop}_"

        for i in range(nfolds):
            modelfile = f"{xprefix}clean_fold{i}.csv"

            # fit the models
            other_xs = [f"{xprefix}X_fold{j}.csv" for j in range(nfolds)
                    if j != i]
            input_files = other_xs
            script = "./CleanInputs.jl"
            target = modelfile
            dependencies = [script, *input_files]
            f.write(f"{target} : {' '.join(dependencies)}\n")
            f.write(f"\t{script} {' '.join(input_files)} --model {target}\n")
            f.write("\n")

            # generate predictions
            for j in range(nfolds):
                script = "./CleanInputs.jl"
                target = f"{xprefix}Xhat{i}_fold{j}.csv"
                input_files = [modelfile, f"{xprefix}X_fold{j}.csv"]
                dependencies = [script, *input_files]
                f.write(f"{target} : {' '.join(dependencies)}\n")
                f.write(f"\t{script} --model {' '.join(input_files)} --output {target}\n")
                f.write("\n")

            # generate predictions for the holdout dataset
            for j in range(nfolds):
                script = "./CleanInputs.jl"
                target = f"{xprefix}Xhat{i}_fold{j}_holdout.csv"
                input_files = [modelfile, f"{xprefix}holdout_X_fold{j}.csv"]
                dependencies = [script, *input_files]
                f.write(f"{target} : {' '.join(dependencies)}\n")
                f.write(f"\t{script} --model {' '.join(input_files)} --output {target}\n")
                f.write("\n")

        # generate a single cleaning model using all of the folds of training data
        if True:
            modelfile = f"{xprefix}clean.csv"
            xs = [f"{xprefix}X_fold{j}.csv" for j in range(nfolds)]

            # fit the models
            input_files = xs
            script = "./CleanInputs.jl"
            target = modelfile
            dependencies = [script, *input_files]
            f.write(f"{target} : {' '.join(dependencies)}\n")
            f.write(f"\t{script} {' '.join(input_files)} --model {target}\n")
            f.write("\n")

            # generate predictions
            script = "./CleanInputs.jl"
            target = f"{xprefix}Xhat.csv"
            input_files = [modelfile] + xs
            dependencies = [script, *input_files]
            f.write(f"{target} : {' '.join(dependencies)}\n")
            f.write(f"\t{script} --model {' '.join(input_files)} --output {target}\n")
            f.write("\n")

            # generate predictions for the holdout dataset
            xs_holdout = [f"{xprefix}holdout_X_fold{j}.csv" for j in range(nfolds)]
            script = "./CleanInputs.jl"
            target = f"{xprefix}Xhat_holdout.csv"
            input_files = [modelfile] + xs_holdout
            dependencies = [script, *input_files]
            f.write(f"{target} : {' '.join(dependencies)}\n")
            f.write(f"\t{script} --model {' '.join(input_files)} --output {target}\n")
            f.write("\n")

        # generate cleaned data for reports
        for i in range(nfolds):
            script = "./CleanInputs.jl"
            target = f"{xprefix}Xhat{i}{reportsuffix}.csv"
            modelfile = f"{xprefix}clean_fold{i}.csv"
            input_files = [modelfile, f"{xprefix}X{reportsuffix}.csv"]
            dependencies = [script, *input_files]
            f.write(f"{target} : {' '.join(dependencies)}\n")
            f.write(f"\t{script} --model {' '.join(input_files)} --output {target}\n")
            f.write("\n")

    # Generate X and Y files for not delirious and no prior delirium cases
    for holdout in ['','_holdout']:
        for xstart,xstop in xtimes:
            xprefix = f"{cachedir}X{xstart}_{xstop}"

            for i in range(nfolds):
                # Generate the rowmasks for no prior delirium case
                script = "./GenerateRowMask.jl"
                target = f"{xprefix}_nopriordelirium_rowmask_fold{i}{holdout}.csv"
                input_files = [f"{xprefix}{holdout}_X_fold{i}.csv"]
                dependencies = [script, *input_files]
                f.write(f"{target} : {' '.join(dependencies)}\n")
                f.write(f"\t{script} {' '.join(input_files)} --column " +
                    f"CAM_alltimemax --removeifgreaterthan 0.0 {target}\n")
                f.write("\n")

                # Generate the rowmasks for not delirious case
                script = "./GenerateRowMask.jl"
                target = f"{xprefix}_notdelirious_rowmask_fold{i}{holdout}.csv"
                input_files = [f"{xprefix}{holdout}_X_fold{i}.csv"]
                dependencies = [script, *input_files]
                f.write(f"{target} : {' '.join(dependencies)}\n")
                f.write(f"\t{script} {' '.join(input_files)} --column " +
                    f"CAM_max --removeifgreaterthan 0.0 {target}\n")
                f.write("\n")

                for c in conditions:
                    rowmask = f"{xprefix}_{c}rowmask_fold{i}{holdout}.csv"

                    # filter the Y files for each of the conditions
                    for ystart,ystop in ytimes:
                        source = f"{cachedir}y{ystart}_{ystop}{holdout}_y_fold{i}.csv"
                        target = f"{xprefix}_y{ystart}_{ystop}_{c}y_fold{i}{holdout}.csv"
                        if c != '':
                            script = "./ApplyRowMask.jl"
                            input_files = [source, rowmask]
                            dependencies = [script, *input_files]
                        else:
                            script = "ln -f"
                            input_files = [source]
                            dependencies = input_files
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\t{script} {' '.join(input_files)} {target}\n")
                        f.write("\n")

                    if c != '':
                        # filter the X files for each of the conditions
                        for j in range(nfolds):
                            script = "./ApplyRowMask.jl"
                            target = f"{xprefix}_{c}Xhat{j}_fold{i}{holdout}.csv"
                            input_files = [f"{xprefix}_Xhat{j}_fold{i}{holdout}.csv",
                                    rowmask]
                            dependencies = [script, *input_files]
                            f.write(f"{target} : {' '.join(dependencies)}\n")
                            f.write(f"\t{script} {' '.join(input_files)} {target}\n")
                            f.write("\n")

            # for the combined datasets
            if True:
                # Generate the rowmasks for no prior delirium case
                script = "./GenerateRowMask.jl"
                target = f"{xprefix}_nopriordelirium_rowmask{holdout}.csv"
                input_files = [f"{xprefix}{holdout}_X_allfolds.csv"]
                dependencies = [script, *input_files]
                f.write(f"{target} : {' '.join(dependencies)}\n")
                f.write(f"\t{script} {' '.join(input_files)} --column " +
                    f"CAM_alltimemax --removeifgreaterthan 0.0 {target}\n")
                f.write("\n")

                # Generate the rowmasks for not delirious case
                script = "./GenerateRowMask.jl"
                target = f"{xprefix}_notdelirious_rowmask{holdout}.csv"
                input_files = [f"{xprefix}{holdout}_X_allfolds.csv"]
                dependencies = [script, *input_files]
                f.write(f"{target} : {' '.join(dependencies)}\n")
                f.write(f"\t{script} {' '.join(input_files)} --column " +
                    f"CAM_max --removeifgreaterthan 0.0 {target}\n")
                f.write("\n")

                for c in conditions:
                    rowmask = f"{xprefix}_{c}rowmask{holdout}.csv"

                    # filter the Y files for each of the conditions
                    for ystart,ystop in ytimes:
                        source = f"{cachedir}y{ystart}_{ystop}{holdout}_y_allfolds.csv"
                        target = f"{xprefix}_y{ystart}_{ystop}_{c}y{holdout}.csv"
                        if c != '':
                            script = "./ApplyRowMask.jl"
                            input_files = [source, rowmask]
                            dependencies = [script, *input_files]
                        else:
                            script = "ln -f"
                            input_files = [source]
                            dependencies = input_files
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\t{script} {' '.join(input_files)} {target}\n")
                        f.write("\n")

                    if c != '':
                        # filter the whole training and holdout set too for each of the conditions
                        script = "./ApplyRowMask.jl"
                        target = f"{xprefix}_{c}Xhat{holdout}.csv"
                        input_files = [f"{xprefix}_Xhat{holdout}.csv",
                                rowmask]
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\t{script} {' '.join(input_files)} {target}\n")
                        f.write("\n")


    for xstart,xstop in xtimes:
        xprefix = f"{cachedir}X{xstart}_{xstop}_"

        # Generate the rowmasks for no prior delirium case for reports
        script = "./GenerateRowMask.jl"
        target = f"{xprefix}nopriordelirium_rowmask{reportsuffix}.csv"
        input_files = [f"{xprefix}X{reportsuffix}.csv"]
        dependencies = [script, *input_files]
        f.write(f"{target} : {' '.join(dependencies)}\n")
        f.write(f"\t{script} {' '.join(input_files)} --column " +
            f"CAM_alltimemax --removeifgreaterthan 0.0 {target}\n")
        f.write("\n")

        # Generate the rowmasks for not delirious case for reports
        script = "./GenerateRowMask.jl"
        target = f"{xprefix}notdelirious_rowmask{reportsuffix}.csv"
        input_files = [f"{xprefix}X{reportsuffix}.csv"]
        dependencies = [script, *input_files]
        f.write(f"{target} : {' '.join(dependencies)}\n")
        f.write(f"\t{script} {' '.join(input_files)} --column " +
            f"CAM_max --removeifgreaterthan 0.0 {target}\n")
        f.write("\n")

        for c in conditions:
            if c != '':
                rowmask = f"{xprefix}{c}rowmask{reportsuffix}.csv"

                # filter the report X files for each of the conditions
                for j in range(nfolds):
                    script = "./ApplyRowMask.jl"
                    target = f"{xprefix}{c}Xhat{j}{reportsuffix}.csv"
                    input_files = [f"{xprefix}Xhat{j}{reportsuffix}.csv",
                            rowmask]
                    dependencies = [script, *input_files]
                    f.write(f"{target} : {' '.join(dependencies)}\n")
                    f.write(f"\t{script} {' '.join(input_files)} {target}\n")
                    f.write("\n")

    for xstart,xstop in xtimes:
        for ystart,ystop in ytimes:
            for c in conditions:
                xprefix = f"{cachedir}X{xstart}_{xstop}_{c}"
                yprefix = f"{cachedir}X{xstart}_{xstop}_y{ystart}_{ystop}_{c}"

                for modelname, modelprefix, modelscript, hasimportance in zip(
                        modelnames, modelprefixes,modelscripts, modelhasimportance):
                    resultnameprefix = (f"X{xstart}_{xstop}_y{ystart}_" +
                        f"{ystop}_{c}{modelprefix}")
                    intermediatefileprefix = f"{cachedir}{resultnameprefix}"
                    outputfileprefix = f"{outputsdir}{resultnameprefix}"

                    # fit the models on the entire training data
                    input_files = [f"{xprefix}Xhat.csv", f"{yprefix}y.csv"]
                    script = modelscript
                    target = f"{intermediatefileprefix}.pickle"
                    dependencies = [script, *input_files]
                    f.write(f"{target} : {' '.join(dependencies)}\n")
                    f.write(f"\tCUDA_VISIBLE_DEVICES=1 {script} train {' '.join(input_files)} {target}\n")
                    f.write("\n")

                    # generate predictions from all of the training data
                    script = modelscript
                    target = f"{intermediatefileprefix}_yhat.csv"
                    input_files = [f"{intermediatefileprefix}.pickle",
                            f"{xprefix}Xhat.csv"]
                    dependencies = [script, *input_files]
                    f.write(f"{target} : {' '.join(dependencies)}\n")
                    f.write(f"\tCUDA_VISIBLE_DEVICES=1 {script} apply {' '.join(input_files)} {target}\n")
                    f.write("\n")

                    # generate predictions from all of the holdout data
                    script = modelscript
                    target = f"{intermediatefileprefix}_yhat_holdout.csv"
                    input_files = [f"{intermediatefileprefix}.pickle",
                            f"{xprefix}Xhat_holdout.csv"]
                    dependencies = [script, *input_files]
                    f.write(f"{target} : {' '.join(dependencies)}\n")
                    f.write(f"\tCUDA_VISIBLE_DEVICES=1 {script} apply {' '.join(input_files)} {target}\n")
                    f.write("\n")

                    # generate importances
                    if hasimportance:
                        script = modelscript
                        target = f"{intermediatefileprefix}_importances.csv"
                        input_files = [f"{intermediatefileprefix}.pickle",
                                f"{xprefix}Xhat_names.json"]
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\t{script} applynames {' '.join(input_files)} " +
                            f"{target}\n")
                        f.write("\n")
                        text_results.append(target)

                    # for every fold
                    for i in range(nfolds):

                        # fit the models on the CV folds
                        other_xs = [f"{xprefix}Xhat{i}_fold{j}.csv" for j in range(nfolds)
                                if j != i]
                        other_ys = [f"{yprefix}y_fold{j}.csv" for j in range(nfolds)
                                if j != i]
                        input_files = [a for b in zip(other_xs, other_ys) for a in b]
                        script = modelscript
                        target = f"{intermediatefileprefix}_fold{i}.pickle"
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\tCUDA_VISIBLE_DEVICES={i} {script} train {' '.join(input_files)} {target}\n")
                        f.write("\n")

                        # generate predictions from training data
                        script = modelscript
                        target = f"{intermediatefileprefix}_yhat_fold{i}.csv"
                        input_files = [f"{intermediatefileprefix}_fold{i}.pickle",
                                f"{xprefix}Xhat{i}_fold{i}.csv"]
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\tCUDA_VISIBLE_DEVICES={i} {script} apply {' '.join(input_files)} {target}\n")
                        f.write("\n")

                        # generate predictions from holdout data
                        script = modelscript
                        target = f"{intermediatefileprefix}_yhat_fold{i}_holdout.csv"
                        input_files = [f"{intermediatefileprefix}_fold{i}.pickle",
                                f"{xprefix}Xhat{i}_fold{i}_holdout.csv"]
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\tCUDA_VISIBLE_DEVICES={i} {script} apply {' '.join(input_files)} {target}\n")
                        f.write("\n")

                        # generate importances
                        if hasimportance:
                            script = modelscript
                            target = f"{intermediatefileprefix}_importances_fold{i}.csv"
                            input_files = [f"{intermediatefileprefix}_fold{i}.pickle",
                                    f"{xprefix}Xhat{i}_names.json"]
                            dependencies = [script, *input_files]
                            f.write(f"{target} : {' '.join(dependencies)}\n")
                            f.write(f"\t{script} applynames {' '.join(input_files)} " +
                                f"{target}\n")
                            f.write("\n")
                            text_results.append(target)

                        # generate predictions for report data
                        script = modelscript
                        target = f"{intermediatefileprefix}_yhat{i}{reportsuffix}.csv"
                        input_files = [f"{intermediatefileprefix}_fold{i}.pickle",
                                f"{xprefix}Xhat{i}{reportsuffix}.csv"]
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\tCUDA_VISIBLE_DEVICES={i} {script} apply {' '.join(input_files)} {target}\n")
                        f.write("\n")

                        # sort the predictions by risk score
                        target = f"{intermediatefileprefix}_yhat{i}{reportsuffix}_sorted.csv"
                        input_files = [f"{intermediatefileprefix}_yhat{i}{reportsuffix}.csv"]
                        dependencies = [*input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\ttail -n +2 {' '.join(input_files)} | sort -r -k 3 -t, >> {target}\n")
                        f.write("\n")

                        for j in range(numpatientreports):
                            # generate a single-row X input file for each of the top-n patients
                            target = f"{intermediatefileprefix}_Xhat{i}{reportsuffix}_risk{j}_oneline.csv"
                            xfile = f"{xprefix}Xhat{i}{reportsuffix}.csv"
                            yfile = f"{intermediatefileprefix}_yhat{i}{reportsuffix}_sorted.csv"
                            input_files = [xfile, yfile]
                            dependencies = [*input_files]
                            f.write(f"{target} : {' '.join(dependencies)}\n")
                            f.write(f"\thead -1 {xfile} > {target}\n"); #colnames
                            f.write(f"\tgrep -h ^`cat {yfile} | cut -d, -f 1-2 | cut -d' ' -f1" +
                                f"| head -{j+1} | tail -1` {xfile} >> {target}\n")
                            f.write("\n")

                            # occlude each of the columns in each single-row input file
                            script = "./OccludeColumns.jl"
                            target = f"{intermediatefileprefix}_Xhat{i}{reportsuffix}_risk{j}_occluded.csv"
                            xfile = f"{intermediatefileprefix}_Xhat{i}{reportsuffix}_risk{j}_oneline.csv"
                            input_files = [xfile]
                            dependencies = [script, *input_files]
                            f.write(f"{target} : {' '.join(dependencies)}\n")
                            f.write(f"\t{script} {' '.join(input_files)} {target}\n")
                            f.write("\n")

                            # generate predictions with the occluded features
                            script = modelscript
                            target = f"{intermediatefileprefix}_yhat{i}{reportsuffix}_risk{j}_occluded.csv"
                            xfile = f"{intermediatefileprefix}_Xhat{i}{reportsuffix}_risk{j}_occluded.csv"
                            input_files = [f"{intermediatefileprefix}_fold{i}.pickle", xfile]
                            dependencies = [script, *input_files]
                            f.write(f"{target} : {' '.join(dependencies)}\n")
                            f.write(f"\tCUDA_VISIBLE_DEVICES={i} {script} apply {' '.join(input_files)} {target}\n")
                            f.write("\n")

                            # generate individual reports from the predictions from occlusions
                            script = "./GenerateOcclusionReport.jl"
                            target = f"{intermediatefileprefix}_report{i}{reportsuffix}_risk{j}.mkd"
                            yfile = f"{intermediatefileprefix}_yhat{i}{reportsuffix}_risk{j}_occluded.csv"
                            xfile = f"{intermediatefileprefix}_Xhat{i}{reportsuffix}_risk{j}_occluded.csv"
                            input_files = [xfile, yfile, "IntervenableFeatures.csv"]
                            dependencies = [script, *input_files]
                            f.write(f"{target} : {' '.join(dependencies)}\n")
                            f.write(f"\t{script} {' '.join(input_files)} {target}\n")
                            f.write("\n")

                        # generate a summary reports from the individual occlusion reports
                        target = f"{outputfileprefix}_report{i}{reportsuffix}_risk.mkd"
                        input_files = [f"{intermediatefileprefix}_report{i}{reportsuffix}_risk{j}.mkd"
                                for j in range(numpatientreports)]
                        dependencies = [*input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\tcat {' '.join(input_files)} > {target}\n")
                        f.write("\n")

                    # create a dummy target for risk scores
                    target = "risk"
                    input_files = [f"{outputfileprefix}_report{i}{reportsuffix}_risk.mkd"
                            for i in range(nfolds)]
                    dependencies = [*input_files]
                    f.write(f"{target} : {' '.join(dependencies)}\n")
                    f.write("\n")

                    # generate plots for cross-validation case...
                    for holdout in ['', '_holdout']:
                        ys = [f"{yprefix}y_fold{j}{holdout}.csv" for j in range(nfolds)]
                        y_hats = [f"{intermediatefileprefix}_yhat_fold{j}{holdout}.csv"
                                for j in range(nfolds)]
                        y_bundle = f"{intermediatefileprefix}_y_folds{holdout}.pickle"
                        y_hat_bundle = f"{intermediatefileprefix}_yhat_folds{holdout}.pickle"

                        # Bundle folds for ys
                        input_files = ys
                        script = "./BundleYFolds.py"
                        target = y_bundle
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\t{script} {' '.join(input_files)} {target}\n")
                        f.write("\n")

                        # Bundle folds for y_hats
                        input_files = y_hats
                        script = "./BundleYFolds.py"
                        target = y_hat_bundle
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\t{script} {' '.join(input_files)} {target}\n")
                        f.write("\n")

                        # ROC plots with cross-validation
                        input_files = [y_bundle, y_hat_bundle]
                        script = "./ROCPlot.py"
                        target = f"{outputfileprefix}_roc{holdout}.png"
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\t{script} --dpi={dpi} " +
                                '--title "" ' +
                                #f"--title \"{modelname} ROC " +
                                #f"({xstart},{xstop})->({ystart},{ystop}))\" " +
                                f"{' '.join(input_files)} {target}\n")
                        f.write("\n")
                        plots.append(target)

                        # PR plots with cross-validation
                        input_files = [y_bundle, y_hat_bundle]
                        script = "./PRPlot.py"
                        target = f"{outputfileprefix}_pr{holdout}.png"
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\t{script} --dpi={dpi} " +
                                '--title "" ' +
                                #f"--title \"{modelname} PR " +
                                #f"({xstart},{xstop})->({ystart},{ystop}))\" " +
                                f"{' '.join(input_files)} {target}\n")
                        f.write("\n")
                        plots.append(target)

                        # calibration plots with cross-validation
                        input_files = [y_bundle, y_hat_bundle]
                        script = "./CalibrationPlot.py"
                        target = f"{outputfileprefix}_calibration{holdout}.png"
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\t{script} --dpi={dpi} " +
                                '--title "" ' +
                                #f"--title \"{modelname} Calibration " +
                                #f"({xstart},{xstop})->({ystart},{ystop}))\" " +
                                f"{' '.join(input_files)} {target}\n")
                        f.write("\n")
                        plots.append(target)

                    # generate plots for bootstrap case...
                    for holdout in ['', '_holdout']:
                        ys = [f"{yprefix}y{holdout}.csv"]
                        y_hats = [f"{intermediatefileprefix}_yhat{holdout}.csv"]
                        y_bundle = f"{intermediatefileprefix}_y{holdout}.pickle"
                        y_hat_bundle = f"{intermediatefileprefix}_yhat{holdout}.pickle"

                        # Bundle folds for ys
                        input_files = ys
                        script = "./BundleYFolds.py"
                        target = y_bundle
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\t{script} {' '.join(input_files)} {target}\n")
                        f.write("\n")

                        # Bundle folds for y_hats
                        input_files = y_hats
                        script = "./BundleYFolds.py"
                        target = y_hat_bundle
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\t{script} {' '.join(input_files)} {target}\n")
                        f.write("\n")

                        # ROC plots with bootstrapping
                        input_files = [y_bundle, y_hat_bundle]
                        script = "./ROCPlot.py"
                        target = f"{outputfileprefix}_roc{holdout}_bootstrap.png"
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\t{script} --dpi={dpi} " +
                                f'--bootstraps {num_bootstraps} ' +
                                '--title "" ' +
                                #f"--title \"{modelname} ROC " +
                                #f"({xstart},{xstop})->({ystart},{ystop}))\" " +
                                f"{' '.join(input_files)} {target}\n")
                        f.write("\n")
                        plots.append(target)

                        # PR plots with cross-validation
                        input_files = [y_bundle, y_hat_bundle]
                        script = "./PRPlot.py"
                        target = f"{outputfileprefix}_pr{holdout}_bootstrap.png"
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\t{script} --dpi={dpi} " +
                                f'--bootstraps {num_bootstraps} ' +
                                '--title "" ' +
                                #f"--title \"{modelname} PR " +
                                #f"({xstart},{xstop})->({ystart},{ystop}))\" " +
                                f"{' '.join(input_files)} {target}\n")
                        f.write("\n")
                        plots.append(target)

                        # calibration plots with cross-validation
                        input_files = [y_bundle, y_hat_bundle]
                        script = "./CalibrationPlot.py"
                        target = f"{outputfileprefix}_calibration{holdout}_bootstrap.png"
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\t{script} --dpi={dpi} " +
                                f'--bootstraps {num_bootstraps} ' +
                                '--title "" ' +
                                #f"--title \"{modelname} Calibration " +
                                #f"({xstart},{xstop})->({ystart},{ystop}))\" " +
                                f"{' '.join(input_files)} {target}\n")
                        f.write("\n")
                        plots.append(target)

                    # generate SHAP plots for the bt case
                    if modelname == "Boosted Trees":
                        # First translate the model to JSON format (so we can
                        # reload it with a newer version of XGBoost)
                        script = "./XgboostPickle2JSON.py"
                        target = f"{intermediatefileprefix}.json"
                        input_files = [f"{intermediatefileprefix}.pickle"]
                        #        f"{xprefix}Xhat_holdout.csv"]
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\t{script} {' '.join(input_files)} {target}\n")
                        f.write("\n")

                        # Next generate the SHAP plot
                        script = "./XgboostShapPlot.py"
                        target = f"{intermediatefileprefix}.json"
                        target = f"{outputfileprefix}_shap_plot.png"
                        input_files = [f"{intermediatefileprefix}.json",
                                f"{xprefix}Xhat_holdout.csv"]
                        dependencies = [script, *input_files, "shap-venv/bin/activate"]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\tshap-venv/bin/python3 {script} --dpi={dpi} " +
                            f"{' '.join(input_files)} {target}\n")
                        f.write("\n")

                        plots.append(target)

                #######################################################
                ### L1 penalty tuning for logistic regression

                loglambdas = np.linspace(-5, 2, 29)
                for loglambda in loglambdas:
                    resultnameprefix = (f"X{xstart}_{xstop}_y{ystart}_" +
                        f"{ystop}_{c}lr{loglambda}")
                    intermediatefileprefix = f"{cachedir}{resultnameprefix}"
                    outputfileprefix = f"{outputsdir}{resultnameprefix}"

                    # for every fold
                    for i in range(nfolds):

                        # fit the models
                        other_xs = [f"{xprefix}Xhat{i}_fold{j}.csv" for j in range(nfolds)
                                if j != i]
                        other_ys = [f"{yprefix}y_fold{j}.csv" for j in range(nfolds)
                                if j != i]
                        input_files = [a for b in zip(other_xs, other_ys) for a in b]
                        script = './LogisticRegression.py'
                        target = f"{intermediatefileprefix}_fold{i}.pickle"
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\tCUDA_VISIBLE_DEVICES={i} {script} train {' '.join(input_files)} {target} --logl1penalty {loglambda}\n")
                        f.write("\n")

                        # generate predictions from training data
                        script = './LogisticRegression.py'
                        target = f"{intermediatefileprefix}_yhat_fold{i}.csv"
                        input_files = [f"{intermediatefileprefix}_fold{i}.pickle",
                                f"{xprefix}Xhat{i}_fold{i}.csv"]
                        dependencies = [script, *input_files]
                        f.write(f"{target} : {' '.join(dependencies)}\n")
                        f.write(f"\tCUDA_VISIBLE_DEVICES={i} {script} apply {' '.join(input_files)} {target}\n")
                        f.write("\n")

                    # AUCs per fold for each value of lambda
                    ys = [f"{yprefix}y_fold{j}.csv" for j in range(nfolds)]
                    y_hats = [f"{intermediatefileprefix}_yhat_fold{j}.csv"
                            for j in range(nfolds)]
                    input_files = [a for b in zip(ys, y_hats) for a in b]
                    script = "./SummarizeAUCs.py"
                    target = f"{intermediatefileprefix}_aucs.txt"
                    dependencies = [script, *input_files]
                    f.write(f"{target} : {' '.join(dependencies)}\n")
                    f.write(f"\t{script} {' '.join(input_files)} {target}\n")
                    f.write("\n")

                # Lambda sweep plot
                resultnameprefix = (f"X{xstart}_{xstop}_y{ystart}_" +
                    f"{ystop}_{c}lr")
                intermediatefileprefix = f"{cachedir}{resultnameprefix}"
                outputfileprefix = f"{outputsdir}{resultnameprefix}"

                input_files = [f'{intermediatefileprefix}{loglambda}_aucs.txt'
                        for loglambda in loglambdas]
                script = "./ParameterSweepPlot.py"
                target = f"{outputfileprefix}_loglambda_sweep.png"
                dependencies = [script, *input_files]
                f.write(f"{target} : {' '.join(dependencies)}\n")
                f.write(f"\t{script} --dpi={dpi} " +
                        '--title "" ' +
                        f"{' '.join([str(loglambda) + ' ' + inputfile for loglambda, inputfile in zip(loglambdas, input_files)])} {target}\n")
                f.write("\n")
                plots.append(target)

    # Generate grid plots
    for xstart,xstop in xtimes:
        for ystart,ystop in ytimes:
            for holdout in ['', '_holdout']:
                y_bundles = [f"{cachedir}X{xstart}_{xstop}_y{ystart}_" +
                    f"{ystop}_{c}{modelprefix}_y_folds{holdout}.pickle"
                    for modelprefix in modelprefixes for c in conditions]
                yhat_bundles = [f"{cachedir}X{xstart}_{xstop}_y{ystart}_" +
                    f"{ystop}_{c}{modelprefix}_yhat_folds{holdout}.pickle"
                    for modelprefix in modelprefixes for c in conditions]
                outputfileprefix = (f"{outputsdir}X{xstart}_{xstop}_y{ystart}_" +
                    f"{ystop}_")
                quoted_rownames = '"' + '" "'.join(modelnames) + '"'
                quoted_colnames = '"' + '" "'.join(conditionnames) + '"'

                # ROC plots
                input_files = y_bundles + yhat_bundles
                script = "./GridPlot.py"
                target = f"{outputfileprefix}_roc_grid{holdout}.png"
                dependencies = [script, *input_files, "ROCPlot.py"]
                f.write(f"{target} : {' '.join(dependencies)}\n")
                f.write(f"\t{script} --dpi={dpi} " +
                        f"--rownames {quoted_rownames} " +
                        f"--colnames {quoted_colnames} " +
                        f"--ys {' '.join(y_bundles)} " +
                        f"--yhats {' '.join(yhat_bundles)} " +
                        f"--out {target}\n")
                f.write("\n")
                plots.append(target)

                # PR plots
                input_files = y_bundles + yhat_bundles
                script = "./GridPlot.py"
                target = f"{outputfileprefix}_pr_grid{holdout}.png"
                dependencies = [script, *input_files, "PRPlot.py"]
                f.write(f"{target} : {' '.join(dependencies)}\n")
                f.write(f"\t{script} --dpi={dpi} " +
                        f"--rownames {quoted_rownames} " +
                        f"--colnames {quoted_colnames} " +
                        f"--ys {' '.join(y_bundles)} " +
                        f"--yhats {' '.join(yhat_bundles)} " +
                        f"--out {target} --precisionrecall\n")
                f.write("\n")
                plots.append(target)

                # calibration plots
                input_files = y_bundles + yhat_bundles
                script = "./GridPlot.py"
                target = f"{outputfileprefix}_calibration_grid{holdout}.png"
                dependencies = [script, *input_files, "CalibrationPlot.py"]
                f.write(f"{target} : {' '.join(dependencies)}\n")
                f.write(f"\t{script} --dpi={dpi} " +
                        f"--rownames {quoted_rownames} " +
                        f"--colnames {quoted_colnames} " +
                        f"--ys {' '.join(y_bundles)} " +
                        f"--yhats {' '.join(yhat_bundles)} " +
                        f"--out {target} --calibration\n")
                f.write("\n")
                plots.append(target)

    # Generate bootstrapped grid plots
    for xstart,xstop in xtimes:
        for ystart,ystop in ytimes:
            for holdout in ['', '_holdout']:
                y_bundles = [f"{cachedir}X{xstart}_{xstop}_y{ystart}_" +
                    f"{ystop}_{c}{modelprefix}_y{holdout}.pickle"
                    for modelprefix in modelprefixes for c in conditions]
                yhat_bundles = [f"{cachedir}X{xstart}_{xstop}_y{ystart}_" +
                    f"{ystop}_{c}{modelprefix}_yhat{holdout}.pickle"
                    for modelprefix in modelprefixes for c in conditions]
                outputfileprefix = (f"{outputsdir}X{xstart}_{xstop}_y{ystart}_" +
                    f"{ystop}_bootstrap")
                quoted_rownames = '"' + '" "'.join(modelnames) + '"'
                quoted_colnames = '"' + '" "'.join(conditionnames) + '"'

                # ROC plots
                input_files = y_bundles + yhat_bundles
                script = "./GridPlot.py"
                target = f"{outputfileprefix}_roc_grid{holdout}.png"
                dependencies = [script, *input_files, "ROCPlot.py"]
                f.write(f"{target} : {' '.join(dependencies)}\n")
                f.write(f"\t{script} --dpi={dpi} " +
                        f"--rownames {quoted_rownames} " +
                        f"--colnames {quoted_colnames} " +
                        f"--bootstraps {num_bootstraps} " +
                        f"--ys {' '.join(y_bundles)} " +
                        f"--yhats {' '.join(yhat_bundles)} " +
                        f"--out {target}\n")
                f.write("\n")
                plots.append(target)

                # PR plots
                input_files = y_bundles + yhat_bundles
                script = "./GridPlot.py"
                target = f"{outputfileprefix}_pr_grid{holdout}.png"
                dependencies = [script, *input_files, "PRPlot.py"]
                f.write(f"{target} : {' '.join(dependencies)}\n")
                f.write(f"\t{script} --dpi={dpi} " +
                        f"--rownames {quoted_rownames} " +
                        f"--colnames {quoted_colnames} " +
                        f"--bootstraps {num_bootstraps} " +
                        f"--ys {' '.join(y_bundles)} " +
                        f"--yhats {' '.join(yhat_bundles)} " +
                        f"--out {target} --precisionrecall\n")
                f.write("\n")
                plots.append(target)

                # calibration plots
                input_files = y_bundles + yhat_bundles
                script = "./GridPlot.py"
                target = f"{outputfileprefix}_calibration_grid{holdout}.png"
                dependencies = [script, *input_files, "CalibrationPlot.py"]
                f.write(f"{target} : {' '.join(dependencies)}\n")
                f.write(f"\t{script} --dpi={dpi} " +
                        f"--rownames {quoted_rownames} " +
                        f"--colnames {quoted_colnames} " +
                        f"--bootstraps {num_bootstraps} " +
                        f"--ys {' '.join(y_bundles)} " +
                        f"--yhats {' '.join(yhat_bundles)} " +
                        f"--out {target} --calibration\n")
                f.write("\n")
                plots.append(target)

    # generate a target for all plots
    f.write(f"all_plots : {' '.join(plots)}\n\n")

    # generate a target for all text results
    f.write(f"all_text_results : {' '.join(text_results)}\n\n")

    # generate a rule for converting from csv to npy files
    f.write(f"%.npy : %.csv\n")
    f.write(f"\t./csv2npy.py $^ $@\n\n")

    # generate a rule for converting from csv to names files
    f.write(f"%_names.json : %_fold0.csv\n")
    f.write(f"\t./csv2featurenames.py $^ $@\n\n")

    # generate a rule for converting from csv to names without folds
    f.write(f"%_names.json : %.csv\n")
    f.write(f"\t./csv2featurenames.py $^ $@\n\n")
