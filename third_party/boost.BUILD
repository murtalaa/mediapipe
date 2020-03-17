# Boost Rules
# Description:
#   Boost libraries for UDP socket programming on Linux

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])


cc_library(
    name = "boost",
    srcs = glob(["lib/libboost*.so*", "lib/libboost*.a*"]),
    hdrs = glob(["include/boost-1_55/**/*.*", "include/boost-1_55/boost/**/*.*"]),
    includes = ["include/boost-1_55/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
