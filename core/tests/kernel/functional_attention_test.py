import logging
import unittest
import torch
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.functional as tkf


class Test(unittest.TestCase):
    def testAttention(self):

        B = tkl.sym.B
        N = tkl.sym.N
        D = tkl.sym.D
        S = tkl.sym.S

        BLOCK_B = tkl.sym.BLOCK_B
        BLOCK_N = tkl.sym.BLOCK_N
        BLOCK_S = tkl.sym.BLOCK_S

        MMA_M = tkl.sym.MMA_M
        MMA_N = tkl.sym.MMA_N
        MMA_K = tkl.sym.MMA_K

        ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
        QUERY_LOAD_ELEMS_PER_THREAD = tkl.sym.QUERY_LOAD_ELEMS_PER_THREAD
        KEY_LOAD_ELEMS_PER_THREAD = tkl.sym.KEY_LOAD_ELEMS_PER_THREAD
        VALUE_LOAD_ELEMS_PER_THREAD = tkl.sym.VALUE_LOAD_ELEMS_PER_THREAD

        STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
        UNROLL_FACTOR = tkl.sym.UNROLL_FACTOR

        constraints = [tkf.WorkgroupConstraint(B, BLOCK_B, 0)]
        constraints = [tkf.WorkgroupConstraint(N, BLOCK_N, 1)]
        constraints += [tkf.TilingConstraint(S, BLOCK_S)]
        constraints += [tkf.WaveConstraint(N, BLOCK_N / 2, 1, 64)]
        constraints += [
            tkf.HardwareConstraint(
                threads_per_wave=64, mma_type="MFMA_F32_16x16x16_F16"
            )
        ]

        @tkf.wave(constraints)
        def attention(
            query: tkl.Memory[N, D, ADDRESS_SPACE, tkl.f16],
            key: tkl.Memory[S, D, ADDRESS_SPACE, tkl.f16],
            value: tkl.Memory[S, D, ADDRESS_SPACE, tkl.f16],
            output: tkl.Memory[N, S, ADDRESS_SPACE, tkl.f32],
        ):
            l_reg = tkf.construct_register_from_metadata((N), tkl.f32, 0.0)
            m_reg = tkf.construct_register_from_metadata((N), tkl.f32, -1e10)
            o_reg = tkf.construct_register_from_metadata((N, D), tkl.f32, 0.0)
            q = tkf.read(query, elements_per_thread=QUERY_LOAD_ELEMS_PER_THREAD)

            @tkf.tiled_loop(S, init_args=[l_reg, m_reg, o_reg])
            def repeat(l_reg, m_reg, o_reg):
                k = tkf.read(key, elements_per_thread=KEY_LOAD_ELEMS_PER_THREAD)
                s = tkf.mma(q, k, transpose_b=True)
                m = tkf.max(s, dim=S)
                p = tkf.exp2(s - m)
                l = tkf.sum(p, dim=S)
                l = tkf.exp2(m_reg - m) * l_reg + l
                v = tkf.read(value, elements_per_thread=VALUE_LOAD_ELEMS_PER_THREAD)
                output = o_reg + tkf.mma(p, v)
                return l, m, output

            l_final, _, o_final = repeat
            o_final = o_final / l_final
            tkf.write(o_final, output, elements_per_thread=STORE_ELEMS_PER_THREAD)

        hyperparams = {
            ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
            QUERY_LOAD_ELEMS_PER_THREAD: 8,
            KEY_LOAD_ELEMS_PER_THREAD: 8,
            VALUE_LOAD_ELEMS_PER_THREAD: 8,
            STORE_ELEMS_PER_THREAD: 1,
            UNROLL_FACTOR: 1,
            BLOCK_B: 1,
            BLOCK_N: 128,
            BLOCK_S: 64,
            MMA_M: 16,
            MMA_N: 16,
            MMA_K: 16,
            B: 48,
            N: 1024,
            D: 64,
            S: 256,
        }
        with tk.gen.TestLaunchContext(hyperparams):
            query = torch.randn(48, 1024, 64, dtype=torch.float16)
            key = torch.randn(48, 256, 64, dtype=torch.float16)
            value = torch.zeros(48, 256, 64, dtype=torch.float16)
            output = torch.zeros(48, 1024, 64, dtype=torch.float32)
            attention(query, key, value, output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
