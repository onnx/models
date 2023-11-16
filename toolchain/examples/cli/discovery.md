# Guiding Model Discovery

This chapter of the `turnkey` CLI tutorial is focused on how to guide the tool as it discovers models. You will learn things such as:
- [How to run model discovery, without spending time on builds or benchmarking](#analyze-only)
- [How to benchmark all the models in all the scripts in a directory](#benchmark-multiple-scripts)
- [How to analyze the building blocks of a model](#maximum-analysis-depth)
- [How to filter which models are passed to the build and benchmark operations](#filtering-model-hashes)

The tutorial chapters are:
1. [Getting Started](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/readme.md)
1. Guiding Model Discovery (this document): `turnkey` arguments that customize the model discovery process to help streamline your workflow.
1. [Working with the Cache](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/cache.md): `turnkey` arguments and commands that help you understand, inspect, and manipulate the `turnkey cache`.
1. [Customizing Builds](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/build.md): `turnkey` arguments that customize build behavior to unlock new workflows.

# Model Discovery Tutorials

All of the tutorials assume that your current working directory is in the same location as this readme file (`examples/cli`).

## Analyze Only

`turnkey` provides the `--analyze-only` argument for when you want to analyze the models in a script, without actually building or benchmarking them.

You can try it out with this command:

```
turnkey benchmark scripts/hello_world.py --analyze-only
```

Which gives a result like:

```
Models discovered during profiling:

hello_world.py:
        pytorch_model (executed 1x - 0.00s)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'hello_world.SmallModel'>)
                Location:       /home/jfowers/onnxmodelzoo/toolchain/examples/cli/scripts/hello_world.py, line 29
                Parameters:     55 (<0.1 MB)
                Hash:           479b1332

pytorch_outputs: tensor([-0.1675,  0.1548, -0.1627,  0.0067,  0.3353], grad_fn=<AddBackward0>)

Woohoo! The 'benchmark' command is complete.
```

You can see that the model is discovered, and some stats are printed, but no build or benchmark took place.

> See the [Analyze Only documentation](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md#analyze-only) for more details.

## Benchmark Multiple Scripts

If you want to benchmark an entire corpus of models, but you don't want to call `turnkey` individually on each model you may provide more than one python file to turnkey at a time.

For example, the command:

```
turnkey scripts/hello_world.py scripts/two_models.py scripts/max_depth.py
```

or the command

```
turnkey scripts/*.py
```

Will iterate over every model in every script in the `scripts` directory, producing a result like this:

```

Models discovered during profiling:

hello_world.py:
        pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'hello_world.SmallModel'>)
                Location:       /home/jfowers/onnxmodelzoo/toolchain/examples/cli/scripts/hello_world.py, line 29
                Parameters:     55 (<0.1 MB)
                Hash:           479b1332
                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                Mean Latency:   0.000   milliseconds (ms)
                                Throughput:     657792.2        inferences per second (IPS)

two_models.py:
        another_pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'two_models.SmallModel'>)
                Location:       /home/jfowers/onnxmodelzoo/toolchain/examples/cli/scripts/two_models.py, line 40
                Parameters:     510 (<0.1 MB)
                Hash:           215ca1e3
                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                Mean Latency:   0.000   milliseconds (ms)
                                Throughput:     509528.6        inferences per second (IPS)

max_depth.py:
        pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          TwoLayerModel (<class 'max_depth.TwoLayerModel'>)
                Location:       /home/jfowers/onnxmodelzoo/toolchain/examples/cli/scripts/max_depth.py, line 41
                Parameters:     85 (<0.1 MB)
                Hash:           80b93950
                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                Mean Latency:   0.000   milliseconds (ms)
                                Throughput:     693955.3        inferences per second (IPS)

pytorch_outputs: tensor([-0.1675,  0.1548, -0.1627,  0.0067,  0.3353], grad_fn=<AddBackward0>)

Woohoo! The 'benchmark' command is complete.
```

You can see that `hello_world.py`, `two_models.py`, and `max_depth.py` are all evaluated.

Alternatively, you can also use `.txt` inputs to list the models you want to benchmark. Text input files may contain regular expressions and may even point to other text files.
To achieve the same result as above, you may simply call `turnkey selected_models.py`, where `selected_models.txt` is:

```text
hello_world.py
two_models.py
max_depth.py
```

> See the [Benchmark Multiple Scripts documentation](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md#benchmark-multiple-scripts) for more details.

## Maximum Analysis Depth

PyTorch models (eg, `torch.nn.Module`) are often built out of a collection of smaller instances. For example, a PyTorch multilayer perceptron (MLP) model may be built out of many `torch.nn.Linear` modules.

Sometimes you will be interested to analyze or benchmark those sub-modules, which is where the `--max-depth` argument comes in.

For example, if you run this command:

```
turnkey benchmark scripts/max_depth.py
```

You will get a result that looks very similar to the [Hello World tutorial](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/readme.md#hello-world) tutorial. However, if you peek into `max_depth.py`, you can see that there are two `torch.nn.Linear` modules that make up the top-level model.

You can analyze and benchmark those `torch.nn.Linear` modules with this command:

```
turnkey benchmark scripts/max_depth.py --max-depth 1
```

You get a result like:

```
Models discovered during profiling:

max_depth.py:
        pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          TwoLayerModel (<class 'max_depth.TwoLayerModel'>)
                Location:       /home/jfowers/onnxmodelzoo/toolchain/examples/cli/scripts/max_depth.py, line 41
                Parameters:     85 (<0.1 MB)
                Hash:           80b93950
                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                Mean Latency:   0.000   milliseconds (ms)
                                Throughput:     533884.4        inferences per second (IPS)

max_depth.py:
                        fc (executed 2x)
                                Model Type:     Pytorch (torch.nn.Module)
                                Class:          Linear (<class 'torch.nn.modules.linear.Linear'>)
                                Parameters:     55 (<0.1 MB)
                                Hash:           6d5eb4ee
                                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                                Mean Latency:   0.000   milliseconds (ms)
                                                Throughput:     809701.4        inferences per second (IPS)

                        fc2 (executed 2x)
                                Model Type:     Pytorch (torch.nn.Module)
                                Class:          Linear (<class 'torch.nn.modules.linear.Linear'>)
                                Parameters:     30 (<0.1 MB)
                                Hash:           d4b2ffa7
                                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                                Mean Latency:   0.000   milliseconds (ms)
                                                Throughput:     677945.2        inferences per second (IPS)
```

You can see that the two instances of `torch.nn.Linear`, `fc` and `fc2`, are benchmarked in addition to the top-level model, `pytorch_model`.

> See the [Max Depth documentation](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md#maximum-analysis-depth) for more details.



## Filtering Model Hashes

When you ran the example from the [Multiple Models per Script](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/readme.md#multiple-models-per-script) tutorial, you saw that `turnkey` discovered, built, and benchmarked two models. What if you only wanted to build and benchmark one of the models?

You can leverage the model hashes feature of `turnkey` to filter which models are acted on. You can see in the result from [Multiple Models per Script](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/readme.md#multiple-models-per-script) that the two models, `pytorch_model` and `another_pytorch_model`, have hashes `479b1332` and `215ca1e3`, respectively.

If you wanted to only build and benchmark `another_pytorch_model`, you could use this command, which filters `two_models.py` with the hash `215ca1e3`:

```
turnkey benchmark scripts/two_models.py::215ca1e3
```

That would produce a result like:

```
Models discovered during profiling:

two_models.py:
        pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'two_models.SmallModel'>)
                Location:       /home/jfowers/onnxmodelzoo/toolchain/examples/cli/scripts/two_models.py, line 32
                Parameters:     55 (<0.1 MB)
                Hash:           479b1332

        another_pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'two_models.SmallModel'>)
                Location:       /home/jfowers/onnxmodelzoo/toolchain/examples/cli/scripts/two_models.py, line 40
                Parameters:     510 (<0.1 MB)
                Hash:           215ca1e3
                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                Mean Latency:   0.000   milliseconds (ms)
                                Throughput:     499272.2        inferences per second (IPS)

pytorch_outputs: tensor([ 0.3628,  0.0489,  0.2952,  0.0021, -0.0161], grad_fn=<AddBackward0>)
more_pytorch_outputs: tensor([-0.1198, -0.5344, -0.1920, -0.1565,  0.2279,  0.6915,  0.8540, -0.2481,
         0.0616, -0.4501], grad_fn=<AddBackward0>)

Woohoo! The 'benchmark' command is complete.
```

You can see that both models are discovered, but only `another_pytorch_model` was built and benchmarked.

> See the [Input Script documentation](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md#input-script) for more details.

## Filtering Model Labels

You can also leverage the labels feature of `turnkey` to filter which models are acted on. Labels are pragmas added by the user to the first line of a `.py` file to list some of the attributes of that given script. `hello_world.py`, for example has the label `test_group::a`, while `two_models.py` and `max_depth.py` have the label `test_group::b`.

If you wanted to only build and benchmark models that have the label `test_group::a`, you could use the command:

```
turnkey scripts/*.py --labels test_group::a
```

That would produce a result like:

```
Models discovered during profiling:

hello_world.py:
        pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'hello_world.SmallModel'>)
                Location:       /home/jfowers/onnxmodelzoo/toolchain/examples/cli/scripts/hello_world.py, line 30
                Parameters:     55 (<0.1 MB)
                Hash:           479b1332
                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                Mean Latency:   0.000   milliseconds (ms)
                                Throughput:     490444.1        inferences per second (IPS)

pytorch_outputs: tensor([-0.1675,  0.1548, -0.1627,  0.0067,  0.3353], grad_fn=<AddBackward0>)

Woohoo! The 'benchmark' command is complete.
```

# Thanks!

Now that you have completed this tutorial, make sure to check out the other tutorials if you want to learn more:
1. [Getting Started](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/readme.md)
1. Guiding Model Discovery (this document): `turnkey` arguments that customize the model discovery process to help streamline your workflow.
1. [Working with the Cache](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/cache.md): `turnkey` arguments and commands that help you understand, inspect, and manipulate the `cache`.
1. [Customizing Builds](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/build.md): `turnkey` arguments that customize build behavior to unlock new workflows.
