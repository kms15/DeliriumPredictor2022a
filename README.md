# DeliriumPrediction2022a

Code for the prediction model described in the manuscript "Daily automated
prediction of delirium risk in hospitalized patients: Model development and
validation" by Kendrick M Shaw, Yu-Ping Shao, Manohar Ghanta, Valdery Moura
Junior, Eyali Y Kimchi, Timothy T Houle, Oluwaseun Akeju, and M. Brandon
Westover.

The data used for training and testing the model contains personally
identifiable health information and thus can not be released with this code;
please contact the authors if access to this data is needed.



## Building the models in a docker container
First, make sure the ImportedData, Cache, and Output directories are clean, then
build the docker container by running

```
docker image build -t delirium2022:latest .
```

Next make sure the patient data is in ImportedData and that the
Outputs/2019-01-17 and the Cache/2019-01-17 directories exist.

Then start an interactive session in the docker container by running the following
command:

```
docker container run \
    --mount type=bind,source=$(pwd)/ImportedData,destination=/src/ImportedData \
    --mount type=bind,source=$(pwd)/Cache,destination=/src/Cache \
    --mount type=bind,source=$(pwd)/Outputs,destination=/src/Outputs \
    --mount type=bind,source=/etc/passwd,destination=/etc/passwd \
    --user $(id -u):$(id -g) \
    --rm -it delirium2022:latest
```

Then you can train the model and generate the plots by running the following
two commands in the container:

```
./GenerateMakefile.py
make
```
