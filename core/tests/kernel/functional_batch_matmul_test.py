import logging
import unittest
import torch
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.functional as tkf


class Test(unittest.TestCase):
    def testGemm(self):

        # Wave tile sizes (determined by constraints below)
        B = tkl.sym.B
        M = tkl.sym.M
        N = tkl.sym.N
        K = tkl.sym.K

        # Workgroup tile sizes
        BLOCK_B = tkl.sym.BLOCK_B
        BLOCK_M = tkl.sym.BLOCK_M
        BLOCK_N = tkl.sym.BLOCK_N
        BLOCK_K = tkl.sym.BLOCK_K
        # MMA tile sizes
        MMA_M = tkl.sym.MMA_M
        MMA_N = tkl.sym.MMA_N
        MMA_K = tkl.sym.MMA_K

        # Address space (for GPU, shared(1) or global(0))
        ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
        # Other hyperparameters
        LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
        STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
        GLOBAL_LOAD_ELEMS_PER_THREAD = tkl.sym.GLOBAL_LOAD_ELEMS_PER_THREAD
        # Unroll factor
        UNROLL_FACTOR = tkl.sym.UNROLL_FACTOR

        # Expose user-constraints
        constraints = [tkf.WorkgroupConstraint(B, BLOCK_B, 0)]
        constraints += [tkf.WorkgroupConstraint(M, BLOCK_M, 0)]
        constraints += [tkf.WorkgroupConstraint(N, BLOCK_N, 1)]
        constraints += [tkf.TilingConstraint(K, BLOCK_K)]
        constraints += [tkf.WaveConstraint(M, BLOCK_M / 2, 0, 64)]
        constraints += [tkf.WaveConstraint(N, BLOCK_N / 2, 1, 64)]
        constraints += [
            tkf.HardwareConstraint(
                threads_per_wave=64, mma_type="MFMA_F32_16x16x16_F16"
            )
        ]

        @tkf.wave(constraints)
        def gemm(
            a: tkl.Memory[B, M, K, ADDRESS_SPACE, tkl.f16],
            b: tkl.Memory[B, N, K, ADDRESS_SPACE, tkl.f16],
            c: tkl.Memory[B, M, N, ADDRESS_SPACE, tkl.f32],
        ):
            c_reg = tkf.construct_register_from_metadata((B, M, N), tkl.f32, 0.0)

            @tkf.tiled_loop(K, init_args=[c_reg])
            def repeat(c_reg) -> tkl.Register[B, M, N, tkl.f32]:
                a_reg = tkf.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                b_reg = tkf.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                c_reg = tkf.mma(a_reg, b_reg, c_reg)
                return c_reg

            tkf.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

        hyperparams = {
            ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 1,
            GLOBAL_LOAD_ELEMS_PER_THREAD: 8,
            UNROLL_FACTOR: 1,
            BLOCK_B: 1,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
            MMA_M: 16,
            MMA_N: 16,
            MMA_K: 16,
            B: 1,
            M: 1024,
            N: 10240,
            K: 1280,
        }
        with tk.gen.TestLaunchContext(hyperparams):
            a = torch.randn(1, 1024, 1280, dtype=torch.float16)
            b = torch.randn(1, 10240, 1280, dtype=torch.float16)
            c = torch.zeros(1, 1024, 10240, dtype=torch.float32)
            gemm(a, b, c)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
