import logging
import unittest
import torch
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.functional as tkf


class Test(unittest.TestCase):
    def testGemm(self):

        # Wave tile sizes (determined by constraints below)
        M = tkl.sym.M
        N = tkl.sym.N
        K = tkl.sym.K

        # Workgroup tile sizes
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

        # Expose user-constraints
        constraints = [tkf.WorkgroupConstraint(M, BLOCK_M, 0)]
        constraints += [tkf.WorkgroupConstraint(N, BLOCK_N, 1)]
        constraints += [tkf.TilingConstraint(K, BLOCK_K)]
        constraints += [tkf.WaveConstraint(M, BLOCK_M / 2, 0, 64)]
        constraints += [tkf.WaveConstraint(N, BLOCK_N / 2, 1, 64)]
        constraints += [
            tkf.HardwareConstraint(
                threads_per_wave=64, mma_type="MFMA_F32_16x16x16_F16"
            )
        ]

        # Wave-level micro-kernel.
        # Since warps are not directly addressable, there is no
        # explicit notion of a warp id (like a workgroup or thread id).
        # Here we use a functional style of expressing the loop since
        # we do not know the loop bounds.
        @tkf.wave(constraints)
        def gemm(
            a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
            b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
            bias_32: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],  # is actually N
            bias_16: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
            c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        ):
            c_reg = tkf.construct_register_from_metadata((M, N), tkl.f32, 0.0)

            @tkf.tiled_loop(K, init_args=[c_reg])
            def repeat(c_reg) -> tkl.Register[M, N, tkl.f32]:
                a_reg = tkf.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                b_reg = tkf.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                c_reg = tkf.mma(a_reg, b_reg, c_reg)
                return c_reg

            b_32 = tkf.read(bias_32, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            biased = repeat + b_32
            b_16 = tkf.read(bias_16, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            result = biased + b_16

            tkf.write(result, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

        hyperparams = {
            ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 1,
            GLOBAL_LOAD_ELEMS_PER_THREAD: 8,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
            MMA_M: 16,
            MMA_N: 16,
            MMA_K: 16,
            M: 128,
            N: 128,
            K: 256,
        }
        with tk.gen.TestLaunchContext(hyperparams):
            a = torch.randn(hyperparams[M], hyperparams[N], dtype=torch.float16)
            b = torch.randn(hyperparams[N], hyperparams[K], dtype=torch.float16)
            bias_32 = torch.randn(
                hyperparams[M], hyperparams[N], dtype=torch.float32
            )  # actually N
            bias_16 = torch.randn(hyperparams[M], hyperparams[N], dtype=torch.float16)
            c = torch.zeros(hyperparams[M], hyperparams[K], dtype=torch.float32)
            gemm(a, b, bias_32, bias_16, c)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
