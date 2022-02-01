licenses(["notice"])  # 3-Clause BSD

exports_files(["license.txt"])

filegroup(
    name = "LICENSE",
    srcs = [
        "license.txt",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkl_headers",
    srcs = glob(["include/*(.cc|.cpp|.cxx|.c++|.C|.c|.h|.hh|.hpp|.ipp|.hxx|.inc|.S|.s|.asm|.a|.lib|.pic.a|.lo|.lo.lib|.pic.lo|.so|.dylib|.dll|.o|.obj|.pic.o)"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkl_libs_linux",
    srcs = [
        "lib/intel64/libiomp5.so",
        "lib/intel64/libmkl_rt.so.1",
        "lib/intel64/libmkl_core.so.1",
        "lib/intel64/libmkl_intel_thread.so.1",
        "lib/intel64/libmkl_intel_lp64.so.1",
        "lib/intel64/libmkl_avx512.so.1",
        "lib/intel64/libmkl_def.so.1",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkl_libs_darwin",
    srcs = [
        "lib/libiomp5.dylib",
        "lib/libmklml.dylib",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkl_libs_windows",
    srcs = [
        "lib/libiomp5md.lib",
        "lib/mklml.lib",
    ],
    linkopts = ["/FORCE:MULTIPLE"],
    visibility = ["//visibility:public"],
)
