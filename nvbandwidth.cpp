/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <argparse/argparse.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvml.h>
#include <iostream>

#ifdef MULTINODE
#include <mpi.h>
#endif

#include "json_output.h"
#include "kernels.cuh"
#include "output.h"
#include "testcase.h"
#include "version.h"
#include "inline_common.h"

int deviceCount;
unsigned int averageLoopCount;
unsigned long long bufferSize;
unsigned long long loopCount;
bool verbose;
bool shouldOutput = true;
bool disableAffinity;
bool skipVerification;
bool useMean;
bool perfFormatter;

Verbosity VERBOSE(verbose);
Verbosity OUTPUT(shouldOutput);

#ifdef MULTINODE
// Device ordinal of the GPU owned by the process
int localRank;
// Process rank within one OS
int localDevice;
int worldRank;
int worldSize;
#endif
char localHostname[STRING_LENGTH];
bool jsonOutput;
Output *output;

// Define testcases here
std::vector<Testcase*> createTestcases() {
    return {
        new HostToDeviceCE(),
        new DeviceToHostCE(),
        new HostToDeviceBidirCE(),
        new DeviceToHostBidirCE(),
        new DeviceToDeviceReadCE(),
        new DeviceToDeviceWriteCE(),
        new DeviceToDeviceBidirReadCE(),
        new DeviceToDeviceBidirWriteCE(),
        new AllToHostCE(),
        new AllToHostBidirCE(),
        new HostToAllCE(),
        new HostToAllBidirCE(),
        new AllToOneWriteCE(),
        new AllToOneReadCE(),
        new OneToAllWriteCE(),
        new OneToAllReadCE(),
        new HostToDeviceSM(),
        new DeviceToHostSM(),
        new HostToDeviceBidirSM(),
        new DeviceToHostBidirSM(),
        new DeviceToDeviceReadSM(),
        new DeviceToDeviceWriteSM(),
        new DeviceToDeviceBidirReadSM(),
        new DeviceToDeviceBidirWriteSM(),
        new AllToHostSM(),
        new AllToHostBidirSM(),
        new HostToAllSM(),
        new HostToAllBidirSM(),
        new AllToOneWriteSM(),
        new AllToOneReadSM(),
        new OneToAllWriteSM(),
        new OneToAllReadSM(),
        new HostDeviceLatencySM(),
        new DeviceToDeviceLatencySM(),
        new DeviceLocalCopy(),
#ifdef MULTINODE
        new MultinodeDeviceToDeviceReadCE(),
        new MultinodeDeviceToDeviceWriteCE(),
        new MultinodeDeviceToDeviceBidirReadCE(),
        new MultinodeDeviceToDeviceBidirWriteCE(),
        new MultinodeDeviceToDeviceReadSM(),
        new MultinodeDeviceToDeviceWriteSM(),
        new MultinodeDeviceToDeviceBidirReadSM(),
        new MultinodeDeviceToDeviceBidirWriteSM(),
        new MultinodeAllToOneWriteSM(),
        new MultinodeAllFromOneReadSM(),
        new MultinodeBroadcastOneToAllSM(),
        new MultinodeBroadcastAllToAllSM(),
        new MultinodeBisectWriteCE(),
#endif
    };
}

Testcase* findTestcase(std::vector<Testcase*> &testcases, std::string id) {
    // Check if testcase ID is index
    char* p;
    long index = strtol(id.c_str(), &p, 10);
    if (*p) {
        // Conversion failed so key is ID
        auto it = find_if(testcases.begin(), testcases.end(), [&id](Testcase* test) {return test->testKey() == id;});
        if (it != testcases.end()) {
            return testcases.at(std::distance(testcases.begin(), it));
        } else {
            throw "Testcase " + id + " not found!";
        }
    } else {
        // ID is index
        if (index < 0 || index >= static_cast<long>(testcases.size())) throw "Testcase index " + id + " out of bound!";
        return testcases.at(index);
    }
}

std::vector<std::string> expandTestcases(std::vector<Testcase*> &testcases, std::vector<std::string> prefixes) {
    std::vector<std::string> testcasesToRun;
    for (auto testcase : testcases) {
         auto it = find_if(prefixes.begin(), prefixes.end(), [&testcase](std::string prefix) {return testcase->testKey().compare(0, prefix.size(), prefix) == 0;});
            if (it != prefixes.end()) {
                testcasesToRun.push_back(testcase->testKey());
            }
    }
    return testcasesToRun;
}

void runTestcase(std::vector<Testcase*> &testcases, const std::string &testcaseID) {
    Testcase* test{nullptr};
    try {
        test = findTestcase(testcases, testcaseID);
    } catch (std::string &s) {
        output->addTestcase(testcaseID, "ERROR", s);
        return;
    }

    try {
        if (!test->filter()) {
            output->addTestcase(test->testKey(), NVB_WAIVED);
            return;
        }

        output->addTestcase(test->testKey(), NVB_RUNNING);

        // Run the testcase
        if (test->testKey() == "host_device_latency_sm" || test->testKey() == "device_to_device_latency_sm") {
            // use fixd-size buffer for latency tests
            test->run(2 * _MiB, loopCount);
        } else {
            test->run(bufferSize * _MiB, loopCount);
        }
    } catch (std::string &s) {
        output->setTestcaseStatusAndAddIfNeeded(test->testKey(), NVB_ERROR_STATUS, s);
    }
}

int main(int argc, char **argv) {
    std::vector<Testcase*> testcases = createTestcases();
    std::vector<std::string> testcasesToRun;
    std::vector<std::string> testcasePrefixes;
    output = new Output();

#ifdef _WIN32
    strncpy(localHostname, getenv("COMPUTERNAME"), STRING_LENGTH - 1);
    const char* computername = getenv("COMPUTERNAME");
    if (computername && computername[0] != '\0') {
        snprintf(localHostname, STRING_LENGTH, "%s", computername);
    } else {
        snprintf(localHostname, STRING_LENGTH, "%s", "unknown");
    }
#else
    ASSERT(0 == gethostname(localHostname, STRING_LENGTH - 1));
#endif
#ifdef MULTINODE
    // Set up MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    // Avoid excessive output by limit output to rank 0
    shouldOutput = (worldRank == 0);
#endif

    // Args parsing
    argparse::ArgumentParser program("nvbandwidth");
    program.add_argument("-h", "--help")
        .help("Produce help message")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("-b", "--bufferSize")
        .help("Memcpy buffer size in MiB")
        .default_value(std::to_string(defaultBufferSize));
    program.add_argument("-l", "--list")
        .help("List available testcases")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("-t", "--testcase")
        .help("Testcase(s) to run (by name or index)")
        .nargs(argparse::nargs_pattern::any);
    program.add_argument("-p", "--testcasePrefixes")
        .help("Testcase(s) to run (by prefix)")
        .nargs(argparse::nargs_pattern::any);
    program.add_argument("-v", "--verbose")
        .help("Verbose output")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("-s", "--skipVerification")
        .help("Skips data verification after copy")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("-d", "--disableAffinity")
        .help("Disable automatic CPU affinity control")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("-i", "--testSamples")
        .help("Iterations of the benchmark")
        .default_value(std::to_string(defaultAverageLoopCount));
    program.add_argument("-m", "--useMean")
        .help("Use mean instead of median for results")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("-j", "--json")
        .help("Print output in json format instead of plain text.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--loopCount")
        .help("Iterations of memcpy to be performed within a test sample")
        .default_value(std::to_string(defaultLoopCount));
    program.add_argument("--perfFormatter")
        .help("Use perf formatter prefix (&&&& PERF) in output")
        .default_value(false)
        .implicit_value(true);

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        output->addVersionInfo();
        std::stringstream errmsg;
        errmsg << "Error parsing command line: " << err.what();
        output->recordError(errmsg.str());
        output->print();
        return 1;
    }

    if (program.get<bool>("--help")) {
        std::cout << program << std::endl;
        return 0;
    }

    // Assign parsed values to variables
    bufferSize = std::stoull(program.get<std::string>("--bufferSize"));
    verbose = program.get<bool>("--verbose");
    skipVerification = program.get<bool>("--skipVerification");
    disableAffinity = program.get<bool>("--disableAffinity");
    useMean = program.get<bool>("--useMean");
    perfFormatter = program.get<bool>("--perfFormatter");
    jsonOutput = program.get<bool>("--json");
    averageLoopCount = std::stoul(program.get<std::string>("--testSamples"));
    loopCount = std::stoull(program.get<std::string>("--loopCount"));
    // Handle multi-token arguments
    if (program.is_used("--testcase")) {
        testcasesToRun = program.get<std::vector<std::string>>("--testcase");
    }
    if (program.is_used("--testcasePrefixes")) {
        testcasePrefixes = program.get<std::vector<std::string>>("--testcasePrefixes");
    }

    output->addVersionInfo();

    if (program.get<bool>("--list")) {
        output->listTestcases(testcases);
        return 0;
    }

    if (testcasePrefixes.size() != 0 && testcasesToRun.size() != 0) {
        output->recordError("You cannot specify both testcase and testcasePrefix options at the same time");
              return 1;
    }


    CU_ASSERT(cuInit(0));
    NVML_ASSERT(nvmlInit());
    CU_ASSERT(cuDeviceGetCount(&deviceCount));
    if (bufferSize < defaultBufferSize) {
        output->recordWarning("NOTE: You have chosen a buffer size that is smaller than the default buffer size. It is suggested to use the default buffer size (64MB) to achieve maximal peak bandwidth.");
    }

    int cudaVersion;
    cudaRuntimeGetVersion(&cudaVersion);

    CU_ASSERT(cuDriverGetVersion(&cudaVersion));

    char driverVersion[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
    NVML_ASSERT(nvmlSystemGetDriverVersion(driverVersion, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE));

    output->addCudaAndDriverInfo(cudaVersion, driverVersion);

    output->recordDevices(deviceCount);

    if (testcasePrefixes.size() > 0) {
        testcasesToRun = expandTestcases(testcases, testcasePrefixes);
        if (testcasesToRun.size() == 0) {
            output->recordError("Specified list of testcase prefixes did not match any testcases");
            return 1;
        }
    }

    // This triggers the loading of all kernels on all devices, even with lazy loading enabled.
    // Some tests can create complex dependencies between devices and function loading requires a
    // device synchronization, so loading in the middle of a test can deadlock.
    preloadKernels(deviceCount);

    if (testcasesToRun.size() == 0) {
        // run all testcases
        for (auto testcase : testcases) {
            runTestcase(testcases, testcase->testKey());
        }
    } else {
        for (const auto& testcaseIndex : testcasesToRun) {
            runTestcase(testcases, testcaseIndex);
        }
    }

    output->print();

    for (auto testcase : testcases) {
        delete testcase;
    }

#ifdef MULTINODE
    MPI_Finalize();
#endif

    output->printInfo();
    return 0;
}
