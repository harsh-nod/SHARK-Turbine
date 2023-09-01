# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

from iree.compiler.ir import (
    Context,
)

from shark_turbine.aot import CompiledModule
from shark_turbine.aot import builtins

import torch


class ApiTest(unittest.TestCase):
    def testBasic(self):
        class BasicModule(CompiledModule):
            ...

        inst = BasicModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn("module @basic", module_str)

    def testExplicitName(self):
        class BasicModule(CompiledModule, export_name="explicit"):
            ...

        inst = BasicModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn("module @explicit", module_str)

    def testJittableFunc(self):
        class BasicModule(CompiledModule):
            @CompiledModule.jittable
            def mul(x, y):
                return x * y

        inst = BasicModule(context=Context())
        self.assertIsInstance(inst.mul, builtins.jittable)

    def testExportedProc(self):
        class ExportedProcModule(CompiledModule):
            def foobar(self):
                ...

        inst = ExportedProcModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)

    def testExportedNoArgJit(self):
        class ExportedProcModule(CompiledModule):
            def foobar(self):
                return self.compute()

            @CompiledModule.jittable
            def compute():
                t1 = torch.ones(2, 2)
                t2 = t1 + t1
                return t2 * t2

        inst = ExportedProcModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
