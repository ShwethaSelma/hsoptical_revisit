﻿# `HSOpticalFlow` Sample
 
The `HSOpticalFlow` is a computation of per-pixel motion estimation between two consecutive image frames caused by movement of object or camera. This sample is implemented using SYCL* by migrating code from original CUDA source code and offloading computations to a CPU, GPU, or accelerator.

| Property                  | Description
|:---                       |:---
| What you will learn       | Migrate and optimize HSOptical sample from CUDA to SYCL.
| Time to complete          | 15 minutes

## Purpose

Optical flow method is based on two assumptions: brightness constancy and spatial flow smoothness. These assumptions are combined in a single energy functional and solution is found as its minimum point. The sample includes both parallel and serial computation, which allows for direct results comparison between CPU and Device.  
The parallel implementation demonstrates the use of key SYCL concepts, such as

- Texture memory
- Cooperative groups
- Shared Memory



 The sample illustrates the steps needed for manual migration of explicit CUDA Graph API's such as cudaGraphCreate(), cudaGraphAddMemcpyNode(), cudaGraphLaunch() to SYCL equivalent API's using [Taskflow](https://github.com/taskflow/taskflow) programming Model.

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

The Jacobi CUDA Graphs sample uses Jacobi iterative algorithm to determines the number of iterations needed to solve system of Linear Equations. All computations happen inside a for-loop.

There are two exit criteria from the loop:
  1.  Execution reaches the maximum number of iterations 
  2.  The final error falls below the desired tolerance.
 
Each iteration has two parts:
  - Jacobi Method computation
  - Final Error computation

In both `Jacobi Method` and `Final Error` Kernels reduction is performed to obtain the final error or sum value. 
The kernel uses cooperative groups, warp-level primitives, atomics and shared memory for the faster and frequent memory access to the block. These computation are loaded into kernel by host function which can be achieved through any one of the three methods. 
  1.  `JacobiMethodGpuCudaGraphExecKernelSetParams()`, which uses explicit CUDA Graph APIs
  2.  `JacobiMethodGpuCudaGraphExecUpdate()`, which uses CUDA stream capture APIs to launch
  3.  `JacobiMethodGpu()`, which uses regular CUDA API's to launch kernels. 
  
  We migrate the first and third host function using SYCLomatic. We then migrate the remaining CUDA Graphs code section using [Taskflow](https://github.com/taskflow/taskflow) Programming Model. 
  We do not migrate `JacobiMethodGpuCudaGraphExecUpdate()`, because CUDA Stream Capture APIs are not yet supported in SYCL.

This sample is migrated from NVIDIA CUDA sample. See the [JacobiCudaGraphs](https://github.com/NVIDIA/cuda-samples/tree/v11.8/Samples/3_CUDA_Features/jacobiCudaGraphs) sample in the NVIDIA/cuda-samples GitHub.

## Prerequisites

| Optimized for              | Description
|:---                        |:---
| OS                         | Ubuntu* 20.04
| Hardware                   | Intel® Gen9, Gen11, and Xeon CPU
| Software                   | SYCLomatic version 2023.1, Intel® oneAPI Base Toolkit version 2023.1

For more information on how to use Syclomatic, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.vmhplg).

## Key Implementation Details

This sample demonstrates the migration of the following prominent CUDA features: 
- CUDA Graph APIs
- CUDA Stream Capture
- Atomic Operations
- Shared memory
- CUDA streams 
- Cooperative groups
- Warp-level Primitives

The Jacobi CUDA Graphs computations happen inside a two- kernel Jacobi Method and Final Error Kernels., Element reduction is performed to obtain the final error or sum value. 
In this sample, the vectors are loaded into shared memory for faster memory access and thread blocks are partitioned into tiles. Then, reduction of input data is performed in each of the partitioned tiles using sub-group primitives. These intermediate results are then added to a final sum variable via an atomic add operation. 

The computation kernels are either scheduled using 2 alternative types of host function calls:

1.  Host function `JacobiMethodGpuCudaGraphExecKernelSetParams()`, which uses explicit CUDA Graph APIs 
2.  Host function `JacobiMethodGpu()`, which uses regular CUDA API's to launch kernels.

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

1.	DPCT1007: Migration of cudaGraphCreate is not supported. 
    ```
    cudaGraphCreate(&graph, 0);
    ```
    Since SYCLomatic Tool doesn’t support migration of CUDA Graphs API yet, we need to manually migrate these APIs with the help of [Taskflow](https://github.com/taskflow/taskflow) programming model which supports SYCL. Taskflow Programming model is a lightweight task graph-based programming model, [tf::syclFlow](https://github.com/taskflow/taskflow/tree/master/taskflow/sycl), for tasking SYCL operations and their dependencies. We need to include the header file, taskflow/sycl/syclflow.hpp, for using tf::syclFlow.
    ```
    tf::Taskflow tflow;
    tf::Executor exe;
    ```
    The first line construct a taskflow graph which is then executed by an executor. 

2.  DPCT1007: Migration of cudaGraphAddMemsetNode is not supported.
    ```
    cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams);
    ```
    The tf::syclFlow::memset method creates a memset task that fills untyped data with a specified byte value. 
    ```
     tf::syclTask dsum_memset = sf.memset(d_sum, 0, sizeof(double)) .name("dsum_memset");
    ```
    For more information on memory operations refer [here](https://github.com/taskflow/taskflow/blob/master/taskflow/sycl/syclflow.hpp).

3.  DPCT1007: Migration of cudaGraphAddKernelNode is not supported.
    ```
     cudaGraphAddKernelNode(&jacobiKernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &NodeParams0);
    ```
    The tf::syclFlow::on creates a task to launch the given command group function object and tf::syclFlow::parallel_for creates a kernel task from a parallel_for method through the handler object associated with a command group. The SYCL runtime schedules command group function objects from out-of-order queue and constructs a task graph based on submitted events.
    ```
    tf::syclTask jM_kernel =
                    sf.on([=](sycl::handler &cgh) {
                         sycl::local_accessor<double, 1> x_shared_acc_ct1(
            		     sycl::range<1>(N_ROWS), cgh);

        	    sycl::local_accessor<double, 1> b_shared_acc_ct1(
           		sycl::range<1>(ROWS_PER_CTA + 1), cgh);
                        cgh.parallel_for(
                            sycl::nd_range<3>(nblocks * nthreads, nthreads),
                            [=](sycl::nd_item<3> item_ct1)
                                [[intel::reqd_sub_group_size(SUB_GRP_SIZE)]] {
                                    JacobiMethod(A, b, conv_threshold, params[k % 2], params[(k + 1) % 2], d_sum, item_ct1,
                           x_shared_acc_ct1.get_pointer(),
                           b_shared_acc_ct1.get_pointer());
				    });
			}).name("jacobi_kernel");
    ```

4.	DPCT1007: Migration of cudaGraphAddMemcpyNode is not supported.
    ```
    cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &memcpyParams);
    ```
    The tf::syclFlow provides memcpy method to creates a memory copy task that copies untyped data in bytes.
    ```
    tf::syclTask sum_d2h = sf.memcpy(&sum, d_sum, sizeof(double)).name("sum_d2h");
    ```  

5.	DPCT1007: Migration of cudaGraphInstantiate is not supported.
    ```
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    ```
    SYCL Task graph doesn’t need to be instantiated before executing but need to establish the task dependencies using precede and succeed.
    ```
    jM_kernel.succeed(dsum_memset).precede(sum_d2h);
    ```
    The jM_kernel will be executed after dsum_memset execution and jM_kernel executes before sum_d2h execution.
	
6.  DPCT1082: Migration of cudaKernelNodeParams type is not supported.
    DPCT1007: Migration of cudaGraphExecKernelNodeSetParams is not supported.
    ```
    cudaKernelNodeParams NodeParams0, NodeParams1;

    cudaGraphExecKernelNodeSetParams(
        graphExec, jacobiKernelNode,
        ((k & 1) == 0) ? &NodeParams0 : &NodeParams1);
    ```
    The tf::syclFlow doesn’t have a equivalent method to cudaGraphExecKernelNodeSetParams method to set the arguments of the kernel conditionally, instead we can achieve this by simple logic by array of arguments and choosing it conditionally.
    ```
    double *params[] = {x, x_new};

    cgh.parallel_for(
                            sycl::nd_range<3>(nblocks * nthreads, nthreads),
                            [=](sycl::nd_item<3> item_ct1)
                                [[intel::reqd_sub_group_size(SUB_GRP_SIZE)]] {
                                    JacobiMethod(A, b, conv_threshold, params[k % 2], params[(k + 1) % 2], d_sum, item_ct1,
                           x_shared_acc_ct1.get_pointer(),
                           b_shared_acc_ct1.get_pointer());
				    });
      
    ```
   
7.  DPCT1007: Migration of cudaGraphLaunch is not supported.
    ```
    cudaGraphLaunch(graphExec, stream);
    ```
    A taskflow graph can be run once or multiple times through an executor using run() method.
    ```
    exe.run(tflow).wait();
    ```   
    To ensure that all the taskflow submissions are completed before calling destructor, we must use wait() during the execution.

8.  DPCT1007: Migration of cudaGraphExecDestroy is not supported.
    ```
    cudaGraphExecDestroy(graphExec);
    ```
    tf::Taskflow class has default destructor operators for both tf::executor and tf::taskflow objects created.
    ```
    ~Executor() 
    ```
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