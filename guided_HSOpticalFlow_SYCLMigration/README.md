# `HSOpticalFlow` Sample
 
The `HSOpticalFlow` is a computation of per-pixel motion estimation between two consecutive image frames caused by movement of object or camera. This sample is implemented using SYCL* by migrating code from original CUDA source code and offloading computations to a CPU, GPU, or accelerator.

| Property                  | Description
|:---                       |:---
| What you will learn       | Migrate and optimize HSOptical sample from CUDA to SYCL.
| Time to complete          | 15 minutes

## Purpose

Optical flow method is based on two assumptions: brightness constancy and spatial flow smoothness. These assumptions are combined in a single energy functional and solution is found as its minimum point. The sample includes both parallel and serial computation, which allows for direct results comparison between CPU and Device. Input images of the sample are computed to get the absolute difference value output(L1 error) between serial and parallel computation. The parallel implementation demonstrates the use of key SYCL concepts, such as

- Image Processing
- SYCL Image memory
- Sub-group primitives 
- Shared Memory

This sample illustrates the steps needed for manual migration of CUDA Texture memory object and API's such as  cudaResourceDesc, cudaTextureDesc, cudaCreateTextureObject() etc, to SYCL equivalent. These CUDA Texture memory API's are manually migrated to SYCL Image memory API's.

> **Note**: We use Intel's open-source SYCLomatic tool which assists developers in porting CUDA code automatically to SYCL code. To finish the process, developers complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. Users can also use the Intel® DPC++ Compatibility Tool, available to augment the Intel® oneAPI Base Toolkit.

This sample contains three versions in the following folders:

| Folder Name                             | Description
|:---                                     |:---
| `01_dpct_output`                        | Contains output of SYCLomatic Tool used to migrate SYCL-compliant code from CUDA code. This SYCL code has some unmigrated code that has to be manually fixed to get full functionality. (The code does not functionally work as supplied.)
| `02_sycl_migrated`                      | Contains manually migrated SYCL code from CUDA code.
| `03_sycl_migrated_optimized`            | Contains manually migrated SYCL code from CUDA code with performance optimizations applied.

### Workflow For CUDA to SYCL migration

Refer [Workflow](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html#gs.s2njvh) for details.

### CUDA source code evaluation

The HSOptical Flow sample includes both serial and parallel implementation of the algorithm in flowGold.cpp and flowCUDA.cu files respectively. In the parallel implementation the computation is distributed among following six kernel,

- `AddKernel()`: Performs vector addition 
- `ComputeDerivativesKernel()`: Computes temporal and spatial derivatives of images 
- `DownscaleKernel()`:Computes image downsizing 
- `JacobiIteration()`: Computes for Jacobi iterarion with border conditions explicitly handled within the kernels 
- `UpscaleKernel()`: Upscales one component of an image displacement field
- `WarpingKernel()`: Warps image with given displacement field 

The host code of downscale, Computederivatives, Upscale and Warping uses texture memory for image data computation. The final computed result of serial and parallel implememtation are then compared based on the threshold value.

This sample is migrated from NVIDIA CUDA sample. See the [HSOpticalFlow](https://github.com/NVIDIA/cuda-samples/tree/v11.8/Samples/5_Domain_Specific/HSOpticalFlow) sample in the NVIDIA/cuda-samples GitHub.

## Prerequisites

| Optimized for              | Description
|:---                        |:---
| OS                         | Ubuntu* 20.04
| Hardware                   | Intel® Gen9, Gen11, and Xeon CPU
| Software                   | SYCLomatic version 2023.1, Intel® oneAPI Base Toolkit version 2023.1

For more information on how to use Syclomatic, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.vmhplg).

## Key Implementation Details

This sample demonstrates the migration of the following prominent CUDA features: 
- CUDA Texture Memory API
- Shared memory
- Cooperative groups

HSOptical flow mainly involves following stages image downscaling and upscaling, image warping, computing derivatives, and computation of jacobi iteration.

Image scaling downscaling or upscaling aims to preserve the visual appearance of the original image when it is resized, without changing the amount of data in that image. An image with a resolution of width × height will be resized to new_width × new_height with a scale factor. A scale factor less than 1 indicates shrinking while a scale factor greater than 1 indicates stretching.

Image warping is a transformation that maps all positions in source image plane to positions in a destination plane. Texture addressing mode is set to Clamp, texture coordinates are unnormalized. Clamp addressing mode to handles the out-of-range coordinates. It eases computing derivatives and warping whenever we need to reflect out-of-range coordinates across borders.

Once the warped image is created, derivatives are computed. For each pixel, the required stencil points from texture are fetched and convolved them with filter kernel. In terms of CUDA, we can create a thread for each pixel. This thread fetches required data and computes derivative.

The next step involves solving for Jacobi iterations. Border conditions are explicitly handled within the kernel. The number of iterations is fixed during computations. This eliminates the need for checking error on every iteration. The required number of iterations can be determined experimentally. To perform one iteration of Jacobi method in a particular point, we need to know results of previous iteration for its four neighbors. If we simply load these values from global memory each value will be loaded four times. We store these values in shared memory. This approach reduces number of global memory accesses, provides better coalescing, and improves overall performance.

Prolongation is performed with bilinear interpolation followed by scaling. and are handled independently. For each output pixel there is a thread that fetches the output value from the texture and scales it.

In CUDA texture memory is used to read and update image data and the equivalent in SYCL is image memory where image objects represent a region of memory managed by the SYCL runtime. The data layout of the image memory is deliberately unspecified to allow implementations to provide a layout optimal to a given device. When accessed on host, image memory may be stored on temporary host memory. When accessed on device, image data is stored in device image memory, which can often be texture memory if the device supports it. In case of Intel integrated graphics there is no dedicated texture memory, so L3 cache is utilized.

## Build the `HSOpticalFlow` Sample for CPU and GPU

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### Tool assisted migration – SYCLomatic 

For this sample, the SYCLomatic Tool automatically migrates ~80% of the CUDA runtime API's to SYCL. Follow these steps to generate the SYCL code using the compatibility tool:

1. git clone https://github.com/NVIDIA/cuda-samples.git
2. cd cuda-samples/Samples/5_Domain_Specific/HSOpticalFlow/
3. Generate a compilation database with intercept-build
   ```
   intercept-build make
   ```
4. The above step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.
5. Pass the JSON file as input to the SYCLomatic Tool. The result is written to a folder named dpct_output. The `--in-root` specifies path to the root of the source tree to be migrated. The `--use-custom-helper` option will make a copy of dpct header files/functions used in migrated code into the dpct_output folder as `include` folder. The `--use-experimental-features` option specifies experimental helper function used to logically group work-items.
   ```
   c2s -p compile_commands.json --in-root ../../.. --use-custom-helper=api --use-experimental-features=logical-group
   ```
   
### Manual workarounds 

The following warnings in the "DPCT1XXX" format are generated by the tool to indicate the code not migrated by the tool and need to be manually modified in order to complete the migration. 

1.	DPCT1059: SYCL only supports 4-channel image format. Adjust the code. 
    ```
    texRes.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    ```
    CUDA HSOptical Flow sample uses single channel image format and SYCL supports only 4 channel image formats. Hence, we need to manually adjust two properties of image data.
    1. Image data type format - Data type of image accessor should be `sycl::float4`.
    2. Image input layout - Image data should be padded for additional image channels.

    Along with these changes we need to adjust [[[[[[[[hhh]]]]]]]
    ```
    tf::Taskflow tflow;
    tf::Executor exe;
    ```
    The first line construct a taskflow graph which is then executed by an executor. 

2.  DPCT1007: Migration of cudaTextureDesc::readMode is not supported.
    ```
    texDescr.readMode = cudaReadModeElementType;
    ```
    The tf::syclFlow::memset method creates a memset task that fills untyped data with a specified byte value. 
    ```
     tf::syclTask dsum_memset = sf.memset(d_sum, 0, sizeof(double)) .name("dsum_memset");
    ```
    For more information on memory operations refer [here](https://github.com/taskflow/taskflow/blob/master/taskflow/sycl/syclflow.hpp).


> **Note**: The SYCL Task Graph Programming Model, syclFlow, leverages the out-of-order property of the SYCL queue to design a simple and efficient scheduling algorithm using topological sort. SYCL can be slower than CUDA graphs because of execution overheads.

### Optimizations

Once the CUDA code is migrated to SYCL successfully and functionality is achieved, we can optimize the code by using profiling tools which helps in identifying the hotspots such as operations/instructions taking longer time to execute, memory utilization etc. 

1.	Reduction operation optimization 
    ```
    for (int offset = item_ct1.get_sub_group().get_local_linear_range() / 2;
         offset > 0; offset /= 2) {
      rowThreadSum += tile32.shuffle_down(rowThreadSum, offset);
    }
    ```
    The sub-group function `shuffle_down` works by exchanging values between work-items in the sub-group via a shift. But needs to be looped to iterate among the sub-groups. 
    ```
    rowThreadSum = sycl::reduce_over_group(tile32, rowThreadSum, sycl::plus<double>());
    ```
    The migrated code snippet with `shuffle_down` API can be replaced with `reduce_over_group` to get better performance. The reduce_over_group implements the generalized sum of the array elements internally by combining values held directly by the work-items in a group. The work-group reduces a number of values equal to the size of the group and each work-item provides one value.

2.	Atomic operation optimization
    ```
     if (item_ct1.get_sub_group().get_local_linear_id() == 0) {
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
          &b_shared[i % (ROWS_PER_CTA + 1)], -rowThreadSum);
    }
    ```
    The `atomic_fetch_add` operation calls automatically add on SYCL atomic object. Here, the atomic_fetch_add is used to sum all the subgroup values into rowThreadSum variable. This can be optimized by replacing the atomic_fetch_add with atomic_ref from sycl namespace.
    ```
    if (tile32.get_local_linear_id() == 0) {
       sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                  sycl:: access::address_space::generic_space>
        at_h_sum{b_shared[i % (ROWS_PER_CTA + 1)]};
        at_h_sum -= rowThreadSum;
    }
    ```
    The `sycl::atomic_ref`, references to value of the object to be added. The result is then assigned to the value of the referenced object.

These optimization changes are performed in JacobiMethod and FinalError Kernels which can be found in `03_sycl_migrated_optimized` folder.

### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```

   By default, this command sequence will build the `02_sycl_migrated` and `03_sycl_migrated_optimized` versions of the program.
   
3. Run the program.
   
   Run `02_sycl_migrated` on GPU.
   ```
   $ make run
   ```   
   Run `02_sycl_migrated` for CPU.
    ```
    $ export ONEAPI_DEVICE_SELECTOR=opencl:cpu
    $ make run
    $ unset ONEAPI_DEVICE_SELECTOR
    ```
    
   Run `03_sycl_migrated_optimized` on GPU.
   ```
   $ make run_smo
   ```   
   Run `03_sycl_migrated_optimized` for CPU.
    ```
    $ export ONEAPI_DEVICE_SELECTOR=opencl:cpu
    $ make run_smo
    $ unset ONEAPI_DEVICE_SELECTOR
    ```
   
#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
$ make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.
  
## Example Output

The following example is for `03_sycl_migrated_optimized` for GPU on **Intel(R) UHD Graphics [0x9a60]**.
```
HSOpticalFlow Starting...

Loading "frame10.ppm" ...
Loading "frame11.ppm" ...
Computing optical flow on CPU...
Computing optical flow on Device...

Processing time on CPU: 1818.056152 (ms)
Processing time on Device: 482.154114 (ms)
L1 error : 0.018193
Built target run_smo
```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
