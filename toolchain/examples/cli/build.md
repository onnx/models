# Customizing Builds

This chapter of the `turnkey` CLI tutorial focuses on techniques to customize the behavior of your `turnkey` builds. You will learn things such as:
- [How to build models without benchmarking them](#build-only)
- [How to customize the build process with Sequences](#sequence-file)

The tutorial chapters are:
1. [Getting Started](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/readme.md)
1. [Guiding Model Discovery](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/discovery.md): `turnkey` CLI arguments that customize the model discovery process to help streamline your workflow.
1. [Working with the Cache](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/cache.md): `turnkey` CLI arguments and commands that help you understand, inspect, and manipulate the `turnkey cache`.
1. Customizing Builds (this document): `turnkey` CLI arguments that customize build behavior to unlock new workflows.

# Build Tutorials

All of the tutorials assume that your current working directory is in the same location as this readme file (`examples/cli`).

## Build Only

`turnkey` provides the `--build-only` argument for when you want to analyze and build the models in a script, without actually benchmarking them.

You can try it out with this command:

```
turnkey benchmark scripts/hello_world.py --build-only
```

Which gives a result like:

```
Models discovered during profiling:

hello_world.py:
        pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'hello_world.SmallModel'>)
                Location:       /home/jfowers/onnxmodelzoo/toolchain/examples/cli/scripts/hello_world.py, line 29
                Parameters:     55 (<0.1 MB)
                Hash:           479b1332
                Status:         Model successfully built!

pytorch_outputs: tensor([-0.1675,  0.1548, -0.1627,  0.0067,  0.3353], grad_fn=<AddBackward0>)

Woohoo! The 'benchmark' command is complete.
```

You can see that the model is discovered and built, but no benchmark took place.

> See the [Build Only documentation](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md#build-only) for more details.

# Thanks!

Now that you have completed this tutorial, make sure to check out the other tutorials if you want to learn more:
1. [Getting Started](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/readme.md)
1. [Guiding Model Discovery](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/discovery.md): `turnkey` arguments that customize the model discovery process to help streamline your workflow.
1. [Working with the Cache](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/cache.md): `turnkey` arguments and commands that help you understand, inspect, and manipulate the `turnkey cache`.
1. Customizing Builds (this document): `turnkey` arguments that customize build behavior to unlock new workflows.