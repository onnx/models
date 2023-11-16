# Learning the `turnkey` CLI

This document is a tutorial for exploring the different features of the `turnkey` command line interface (CLI). You can learn the details of those features in the [Tools User Guide](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md) and learn about their implementation in the [Code Organization Guide](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/code.md).

We've created this tutorial document because `turnkey` is a CLI that benchmarks the contents of `.py` scripts. So all of the `.py` scripts in the `examples/cli/scripts` directory are meant to be fed into `turnkey` to demonstrate some specific functionality.

Once you've familiarized yourself with these features, head over to the [`models` directory](https://github.com/aig-bench/onnxmodelzoo/tree/main/models) to learn how to use `turnkey` with real world machine learning models.

The tutorials are organized into a few chapters:
1. Getting Started (this document)
1. [Guiding Model Discovery](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/discovery.md): `turnkey` arguments that customize the model discovery process to help streamline your workflow.
1. [Working with the Cache](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/cache.md): `turnkey` arguments and commands that help you understand, inspect, and manipulate the `cache`.
1. [Customizing Builds](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/build.md): `turnkey` arguments that customize build behavior to unlock new workflows.

In this tutorial you will learn things such as:
- [How to benchmark BERT with one command](#just-benchmark-bert)
- [A "hello world" example, which is the easiest way to get started](#hello-world)
- [Benchmarking on Nvidia GPUs](#nvidia-benchmarking)
- [Working with scripts that invoke more than one model](#multiple-models-per-script)
- [Working with scripts that invoke a model multiple times](#multiple-invocations-of-a-model)
- [Benchmarking an ONNX file](#onnx-benchmarking)

# Just Benchmark BERT

A fun way to get started with `turnkey` is to simply benchmark the popular [BERT transformer model](https://huggingface.co/docs/transformers/model_doc/bert) with a single command:

```
models=$(turnkey models location --quiet)
turnkey $models/transformers/bert.py
```

> _Note_: You will need to [install the models benchmark requirements](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/install.md#turnkeyml-models-requirements), if you haven't already.

This will produce a result that looks like this, which shows you the performance of BERT-Base on your CPU:

```
Models discovered during profiling:

bert.py:
        model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          BertModel (<class 'transformers.models.bert.modeling_bert.BertModel'>)
                Location:       /home/jfowers/onnxmodelzoo/toolchain/models/transformers/bert.py, line 18
                Parameters:     109,482,240 (208.8 MB)
                Hash:           d59172a2
                Status:         Successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz (ort v1.14.1)
                                Mean Latency:   345.341 milliseconds (ms)
                                Throughput:     2.9     inferences per second (IPS)
```


# Tutorials

All of the following tutorials assume that your current working directory is in the same location as this readme file (`examples/cli`).

These tutorials assume you have used the [cloning install](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/install.md#cloning-install) since that provides you with the required tutorial files in `examples/cli/scripts`.

## Hello World

We can perform a basic invocation of `turnkey` to benchmark a PyTorch model by simply running the following command:

```
turnkey scripts/hello_world.py
```

That commands `turnkey` benchmark `hello_world.py` on your CPU. Specifically, `turnkey` takes the following actions:
1. Pass `scripts/hello_world.py` as the input_script to the `benchmark` command of `turnkey`.
  - _Note_: `turnkey <.py file>` is a shorthand for `turnkey benchmark <.py file>`.
1. Run `hello_world.py` against a profiler and look for models from supported machine learning frameworks (e.g. Pytorch).
1. Discover the `pytorch_model` instance of class `SmallModel`, which is a PyTorch model, and print some statistics about it.
1. Export `pytorch_model` to an ONNX file, optimize that ONNX file, and convert it to the `float16` data type.
1. Measure the performance of the ONNX file on your x86 CPU and report the `mean latency` and `throughput`.

The result looks like this:

```
Models discovered during profiling:

hello_world.py:
        pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'hello_world.SmallModel'>)
                Location:       /home/jfowers/onnxmodelzoo/toolchain/examples/cli/hello_world.py, line 29
                Parameters:     55 (<0.1 MB)
                Hash:           479b1332
                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                Mean Latency:   0.001   milliseconds (ms)
                                Throughput:     185964.8        inferences per second (IPS)

pytorch_outputs: tensor([-0.1675,  0.1548, -0.1627,  0.0067,  0.3353], grad_fn=<AddBackward0>)

Woohoo! The 'benchmark' command is complete.
```

You can see on the `Status:` line that the `pytorch_model` was benchmarked on a `Intel(R) Xeon(R) CPU @ 2.20GHz` x86 device and that the mean latency and throughput are both reported.

## Nvidia Benchmarking

By default, `turnkey` uses x86 CPUs for benchmarking, however benchmarking on Nvidia GPUs is also supported using the `--device` argument.

If you have an Nvidia GPU installed on your machine, you can benchmark `hello_world.py` by running the following command:

```
turnkey scripts/hello_world.py --device nvidia
```

To get a result like this:

```
Models discovered during profiling:

hello_world.py:
	pytorch_model (executed 1x)
		Model Type:	Pytorch (torch.nn.Module)
		Class:		SmallModel (<class 'hello_world.SmallModel'>)
		Location:	/home/jfowers/onnxmodelzoo/toolchain/examples/cli/hello_world.py, line 29
		Parameters:	55 (<0.1 MB)
		Hash:		479b1332
		Status:		Model successfully benchmarked on NVIDIA A100-SXM4-40GB
				Mean Latency:	0.027	milliseconds (ms)
				Throughput:	21920.5	inferences per second (IPS)

pytorch_outputs: tensor([-0.1675,  0.1548, -0.1627,  0.0067,  0.3353], grad_fn=<AddBackward0>)

Woohoo! The 'benchmark' command is complete.
```

You can see that the device mentioned in the status is a `NVIDIA A100-SXM4-40GB`.

## Multiple Models per Script

The tools will benchmark all models discovered in the input script. We can demonstrate this with the `two_models.py` script.

Run the following command:

```
turnkey scripts/two_models.py
```

To get a result like:

```
Models discovered during profiling:

two_models.py:
        pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'two_models.SmallModel'>)
                Location:       /home/jfowers/onnxmodelzoo/toolchain/examples/cli/scripts/two_models.py, line 32
                Parameters:     55 (<0.1 MB)
                Hash:           479b1332
                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                Mean Latency:   0.000   milliseconds (ms)
                                Throughput:     640717.1        inferences per second (IPS)

        another_pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'two_models.SmallModel'>)
                Location:       /home/jfowers/onnxmodelzoo/toolchain/examples/cli/scripts/two_models.py, line 40
                Parameters:     510 (<0.1 MB)
                Hash:           215ca1e3
                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                Mean Latency:   0.000   milliseconds (ms)
                                Throughput:     642021.1        inferences per second (IPS)

pytorch_outputs: tensor([ 0.3628,  0.0489,  0.2952,  0.0021, -0.0161], grad_fn=<AddBackward0>)
more_pytorch_outputs: tensor([-0.1198, -0.5344, -0.1920, -0.1565,  0.2279,  0.6915,  0.8540, -0.2481,
         0.0616, -0.4501], grad_fn=<AddBackward0>)

Woohoo! The 'benchmark' command is complete.
```

You can see that both model instances in `two_models.py`, `pytorch_model` and `another_pytorch_model`, are both discovered and benchmarked.

## Multiple Invocations of a Model

A single script may invoke the same model multiple times using different input shapes (e.g. when varying the batch size). When this happens, the tools will benchmark and display each of those invocations as sub-results of the same model instance. 

> **Note**: multiple invocations of a model with the same input shape will only be benchmarked once.

The `multiple_invocations.py` script instantiates a single model and invokes it three times. The fist two times the model is invoked with inputs of the same shape (batch 1), while the third invocation uses a different input shape (batch 2). Note that two unique static model invocations are identified. 

Run the following command:

```
turnkey scripts/multiple_invocations.py
```

To get a result like:
```
Models discovered during profiling:

multiple_invocations.py:
        pytorch_model
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'multiple_invocations.SmallModel'>)
                Location:       /home/dhnoronha/onnxmodelzoo/toolchain/examples/cli/scripts/multiple_invocations.py, line 40
                Parameters:     60 (<0.1 MB)

                With input shape 1 (executed 2x)
                Input Shape:    'x': (1, 11)
                Hash:           b4aa73ae
                Status:         Successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz (ort v1.14.1)
                                Mean Latency:   0.013   milliseconds (ms)
                                Throughput:     77909.6 inferences per second (IPS)

                With input shape 2 (executed 1x)
                Input Shape:    'x': (2, 11)
                Hash:           cfaa2e2c
                Status:         Successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz (ort v1.14.1)
                                Mean Latency:   0.015   milliseconds (ms)
                                Throughput:     64938.1 inferences per second (IPS)
```

## ONNX Benchmarking

If you already happen to have an ONNX file, `turnkey` can benchmark it for you. We can demonstrate this with the ONNX file in `examples/cli/onnx/sample.onnx`.

Run the following command:

```
turnkey onnx/sample.onnx
```

To get a result like:

```
Building "sample"
    ✓ Receiving ONNX Model   
    ✓ Finishing up   

Woohoo! Saved to ~/onnxmodelzoo/toolchain/examples/cli/onnx/tmp_cache/sample

Info: Benchmarking on local x86...

Info: Performance of build sample on x86 device Intel(R) Xeon(R) CPU @ 2.20GHz is:
        Mean Latency: 0.042 milliseconds (ms)
        Throughput: 23921.9 inferences per second (IPS)
```

# Thanks!

Now that you have completed this tutorial, make sure to check out the other tutorials if you want to learn more:
1. [Guiding Model Discovery](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/discovery.md): `turnkey` arguments that customize the model discovery process to help streamline your workflow.
1. [Working with the Cache](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/cache.md): `turnkey` arguments and commands that help you understand, inspect, and manipulate the `turnkey cache`.
1. [Customizing Builds](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/cache.md): `turnkey` arguments that customize build behavior to unlock new workflows.