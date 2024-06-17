import logging
import unittest
import torch
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.functional as tkf


class Test(unittest.TestCase):
    def testArgmax(self):

        M = tkl.sym.M
        N = tkl.sym.N
        BLOCK_M = tkl.sym.BLOCK_M

        ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
        LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
        STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
        UNROLL_FACTOR = tkl.sym.UNROLL_FACTOR

        constraints = [tkf.WorkgroupConstraint(M, BLOCK_M, 0)]
        constraints += [tkf.WaveConstraint(M, BLOCK_M / 2, 1, 64)]
        constraints += [tkf.HardwareConstraint(threads_per_wave=64)]

        @tkf.wave(constraints)
        def argmax(
            input: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
            output: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
        ):
            input_tile = tkf.read(input, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            indices = tkf.iota((M, N))
            input_with_indices = tkf.concat(
                [input_tile, indices], desired_shape=(2, M, N)
            )
            max_with_indices = tkf.max(input_with_indices, dim=N)
            tkf.write(
                max_with_indices[1], output, elements_pre_thread=STORE_ELEMS_PER_THREAD
            )
            return indices

        hyperparams = {
            ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
            LOAD_ELEMS_PER_THREAD: 8,
            STORE_ELEMS_PER_THREAD: 1,
            UNROLL_FACTOR: 1,
            BLOCK_M: 128,
            M: 1024,
            N: 5120,
        }
        with tk.gen.TestLaunchContext(hyperparams):
            input = torch.randn(1024, 5120, dtype=torch.float16)
            output = torch.zeros(1024, dtype=torch.float16)
            argmax(input, output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
