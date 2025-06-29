# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "DialEgg"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "Inputs",
    "Examples",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
] + [
    "3mm",
    "arith",
    "classic",
    "constant_folding",
    "image_conversion",
    "nmm",
    "polynomial",
    "scf",
    "util",
    "vector_norm",
]

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.dialegg_obj_root, "test")
config.dialegg_tools_dir = config.dialegg_obj_root
config.dialegg_libs_dir = os.path.join(config.dialegg_obj_root, "lib")

config.substitutions.append(("%dialegg_libs", config.dialegg_libs_dir))
config.substitutions.append(
    ("%eggopt", os.path.join(config.dialegg_tools_dir, "egg-opt"))
)
config.substitutions.append(("%testsrc", config.test_source_root))

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.dialegg_tools_dir, config.llvm_tools_dir]
tools = ["egg-opt"]

llvm_config.add_tool_substitutions(tools, tool_dirs)

llvm_config.with_environment(
    "PYTHONPATH",
    [
        os.path.join(config.mlir_obj_dir, "python_packages", "dialegg"),
    ],
    append_path=True,
)
