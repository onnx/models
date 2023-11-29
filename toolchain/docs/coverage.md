# Evaluating TurnkeyML Coverage

This is a simple tutorial on how to evaluate TurnkeyML test coverage using [Coverage.py](https://coverage.readthedocs.io/en/7.3.2/#quick-start).

## Basic Test Coverage

### Installation

On your main `tkml` environment, run `pip install coverage`.

### Gathering Results

To gather results, cd into the test folder on `toolchain\test` and call `coverage run` on each of the tests as shown below.

```
coverage run --data-file=.coverage_unit -m unittest unit.py
coverage run --data-file=.coverage_analysis -m unittest analysis.py
coverage run --data-file=.coverage_build_model -m unittest build_model.py
coverage run --data-file=.coverage_model_api -m unittest model_api.py
coverage run --data-file=.coverage_cli -m unittest cli.py
```

### Combining Results

You can the combine all results into a single file using the command shown below.

```
coverage combine --keep .coverage_analysis .coverage_build_model .coverage_cli .coverage_model_api .coverage_ryzen .coverage_unit
```

This will generate a combined file called `.coverage`.

### Generating Report

For a human-readable report, run `coverage html -i --data-file=.coverage` to generate an html report. If your goal is to later read this information programmatically, `coverage json -i --data-file=.coverage` is a better option.

Below in an example the type of information you will find in those reports:

```
Name                                      Stmts   Miss Branch BrPart  Cover   Missing
--------------------------------------------------------------------------------------------------------
\turnkeyml\build\__init__.py                 0      0      0      0   100%
\turnkeyml\build\onnx_helpers.py            70     34     28      2    45%   15-21, 28-87, 92, 95-100
\turnkeyml\build\quantization_helpers.py    29     20     18      0    19%   13-30, 35, 50-78
\turnkeyml\build\sequences.py               15      1      8      2    87%   62->61, 65
\turnkeyml\build\tensor_helpers.py          47     26     34      4    41%   17-44, 57, 61, 63-74, 78
\turnkeyml\build_api.py                     31      9      8      3    64%   68-71, 120-125, 140-147
\turnkeyml\cli\__init__.py                   0      0      0      0   100%
...
--------------------------------------------------------------------------------------------------------
TOTAL                                     4145   1329   1344    176    63%      
```

## Advanced Test Coverage

TurnkeyML spawns sub-processes in multiple scenarios to do things such as enabling the use of multiple conda environments. Sub-processes are also spawned in many of our tests.

Measuring coverage in those sub-processes can be tricky because we have to modify the code spawning the process to invoke `coverage.py`.

Enabling tracing of coverage on sub-processes is currently only partially possible, as some of the subprocesses used inside `turnkey` fail when used with `coverage.py`.

The instructions below show how to measure coverage using this advanced setup.

Please note that, without this advanced setup, files like `within_conda.py` are not analyzed at all. This happens because the execution of files like `within_conda.py` happen within a subprocess.

### Preparation

#### Step 1: Installing coverage on all environments

First, make sure to `pip install coverage` on all environments used by `turnkey`. Run `conda env list` and install `coverage` on all environments that are named `turnkey-onnxruntime-*-ep`. From now on, we will refer to those as `turnkey environments`.

#### Step 2: Edit Python processes startup

Now, we have to configure Python to invoke `coverage.process_startup()` when Python processes start. To do this, add a file named `sitecustomize.py` to `<YOUR_PATH>\miniconda3\envs\<turnkey-onnxruntime-*-ep>\Lib\site-packages\sitecustomize.py`, where `<turnkey-onnxruntime-*-ep>` corresponds to each of your turnkey environments. Each of those files should have the content shown below:

```python
import coverage
coverage.process_startup()
print("STARTING COVERAGE MODULE")
```


### Gathering Data and Generating Reports

To gather data and generate reports, simply follow the instructions provided in the previous section.