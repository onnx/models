# Working with the Cache

This chapter of the `turnkey` CLI tutorials is focused on understanding, inspecting, and manipulating the `turnkey cache`. You will learn things such as:
- [How to change the cache directory for a `turnkey benchmark` run](#cache-directory)
- [How to list all the builds in a cache](#cache-list-command)
- [How to get statistics about a build in a cache](#cache-stats-command)
- [How to delete a build from a cache](#cache-delete-command)
- [How to change the change the cache directory for `turnkey cache` commands](#cache-commands-with---cache-dir)
- [How to keep your filesystem from filling up with build artifacts](#lean-cache)

The tutorial chapters are:
1. [Getting Started](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/readme.md)
1. [Guiding Model Discovery](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/discovery.md): `turnkey` arguments that customize the model discovery process to help streamline your workflow.
1. Working with the Cache (this document): `turnkey` arguments and commands that help you understand, inspect, and manipulate the `turnkey cache`.
1. [Customizing Builds](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/build.md): `turnkey` arguments that customize build behavior to unlock new workflows.

# Cache Tutorials

All of the tutorials assume that your current working directory is in the same location as this readme file (`examples/cli`).

## Cache Directory

By default, the tools use `~/.cache/turnkey/` as the location for the cache (see the [Build documentation](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md#build) for more details).

However, you might want to set the cache location for any number of reasons. For example, you might want to keep the results from benchmarking one corpus of models separate from the results from another corpus.

You can try this out with the following command:

```
turnkey benchmark scripts/hello_world.py --cache-dir tmp_cache
```

When that command completes, you can use the `ls` command to see that `tmp_cache` has been created at your command line location. 

See the Cache Commands tutorials below to see what you can do with the cache.

> See the [Cache Directory documentation](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md#cache-directory) for more details.

## Cache List Command

This tutorial assumes you have completed the [Cache Directory](#cache-directory) and [Benchmark Multiple Scripts documentation](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md#benchmark-multiple-scripts) tutorials, and that the `tmp_cache` directory exists at your command line location.

You can use the `cache list` command to see what builds are available in your cache:

```
turnkey cache list
```

Which produces a result like:

```
Info: Builds available in cache ~/.cache/turnkey:
hello_world_479b1332     max_depth_80b93950       two_models_479b1332      
two_models_215ca1e3 
```

## Cache Stats Command

This tutorial assumes you have completed the prior tutorials in this document.

You can learn more about a build with the `cache stats` command:

```
turnkey cache stats hello_world_479b1332
```

Which will print out a lot of statistics about the build, like:

```
Info: The state of build hello_world_479b1332 in cache ~/.cache/turnkey is:
build_status: successful_build
cache_dir: /home/jfowers/.cache/turnkey
config:
...
```

## Cache Delete Command

This tutorial assumes you have completed the prior tutorials in this document.

You can also delete a build from a cache with the `cache delete` command. Be careful, this permanently deletes the build!

For example, you could run the commands:

```
turnkey cache delete max_depth_80b93950
turnkey cache list
```

And you would see that the cache no longer includes the build for `max_depth`:

```
Info: Builds available in cache ~/.cache/turnkey:
hello_world_479b1332     two_models_215ca1e3      two_models_479b1332 
```

## Cache Commands with --cache-dir

This tutorial assumes you have completed the prior tutorials in this document.

Finally, the `cache` commands all take a `--cache-dir` that allows them to operate on a specific cache directory (see the [Cache Directory tutorial](#cache-directory) for more details).

For example, you can run this command:

```
turnkey cache list --cache-dir tmp_cache
```

Which will produce this result, if you did the [Cache Directory tutorial](#cache-directory):

```
Info: Builds available in cache tmp_cache:
hello_world_479b1332  
```

> See the [Cache Commands documentation](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md#cache-commands) for more details.

## Lean Cache

As you progress, you may notice that the cache directory can take up a lot of space on your hard disk because it produces a lot of ONNX files and other build artifacts. 

We provide the `--lean-cache` option to help in the situation where you want to collect benchmark results, but you don't care about keeping the build artifacts around.

First, do the [Cache Directory tutorial](#cache-directory) so that we have a nice convenient cache directory to look at.

Next, run the command:

```
ls -shl tmp_cache/hello_world_479b1332
```

to see the contents of the cache directory, along with their file sizes:

```
total 32K
4.0K drwxr-xr-x 2 jfowers 4.0K Feb 16 08:14 compile
4.0K -rw-r--r-- 1 jfowers 2.0K Feb 16 08:14 hello_world_479b1332_state.yaml
4.0K -rw-r--r-- 1 jfowers  396 Feb 16 08:14 inputs_original.npy
4.0K -rw-r--r-- 1 jfowers   84 Feb 16 08:14 log_export_pytorch.txt
4.0K -rw-r--r-- 1 jfowers   71 Feb 16 08:14 log_fp16_conversion.txt
4.0K -rw-r--r-- 1 jfowers   63 Feb 16 08:14 log_optimize_onnx.txt
   0 -rw-r--r-- 1 jfowers    0 Feb 16 08:14 log_set_success.txt
4.0K drwxr-xr-x 2 jfowers 4.0K Feb 16 08:14 onnx
4.0K drwxr-xr-x 3 jfowers 4.0K Feb 16 08:14 x86_benchmark
```

These file sizes aren't too bad because the `pytorch_model` in the [Cache Directory tutorial](#cache-directory) isn't very large. But imagine if you were using GPT-J 6.7B instead, there would be tens of gigabytes of data left on your disk.

Now run the following command to repeat the [Cache Directory tutorial](#cache-directory) in lean cache mode:

```
turnkey benchmark scripts/hello_world.py --cache-dir tmp_cache --lean-cache
```

And then inspect the build directory again:

```
ls -shl tmp_cache/hello_world_479b1332
```

To see that the `onnx` and `x86_benchmark` directories are gone, thereby saving disk space:

```
total 20K
4.0K drwxr-xr-x 2 jfowers 4.0K Feb 16 08:14 compile
4.0K -rw-r--r-- 1 jfowers 2.0K Feb 16 08:14 hello_world_479b1332_state.yaml
4.0K -rw-r--r-- 1 jfowers   84 Feb 16 08:14 log_export_pytorch.txt
4.0K -rw-r--r-- 1 jfowers   71 Feb 16 08:14 log_fp16_conversion.txt
4.0K -rw-r--r-- 1 jfowers   63 Feb 16 08:14 log_optimize_onnx.txt
   0 -rw-r--r-- 1 jfowers    0 Feb 16 08:14 log_set_success.txt
```

> See the [Lean Cache documentation](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md#lean-cache) for more details.

> _Note_: If you want to get rid of build artifacts after the build is done, you can run `turnkey cache clean build_name`.

# Thanks!

Now that you have completed this tutorial, make sure to check out the other tutorials if you want to learn more:
1. [Getting Started](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/readme.md)
1. [Guiding Model Discovery](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/discovery.md): `turnkey` arguments that customize the model discovery process to help streamline your workflow.
1. Working with the Cache (this document): `turnkey` arguments and commands that help you understand, inspect, and manipulate the `turnkey cache`.
1. [Customizing Builds](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/build.md): `turnkey` arguments that customize build behavior to unlock new workflows.