#!/usr/bin/env python3
import re

def find_ssa_value(file_name, arg_name):
    with open(file_name, 'r') as f:
        for line in f.readlines():
            if 'stream.binding.subspan' in line and arg_name in line:
                matches = re.match(r'(.*) = stream.binding.subspan', line)
                return matches.group(1).strip()

def replace_in_file(data, search_text, replace_text):
    # Replace the target string
    new_data = data.replace(search_text, replace_text)
    print(f"Replaced '{search_text}' with '{replace_text}'")
    return new_data

def convert_mm_to_bmm(file_name, B, M, N, K, input_dtype, output_dtype):
    A_ssa_val = find_ssa_value(file_name, 'arg0')[1:]
    print("A_ssa_value = ", A_ssa_val)
    C_ssa_val = find_ssa_value(file_name, 'arg2')[1:]
    print("C_ssa_value = ", C_ssa_val)
    replacements: list[tuple[str, str]] = [
        # A type
        (
            f"memref<{M}x{K}x{input_dtype}, strided<[{K}, 1]",
            f"memref<{B}x{M}x{K}x{input_dtype}, strided<[{M*K}, {K}, 1]",
        ),
        # reading from A
        (f"%{A_ssa_val}[", f"%{A_ssa_val}[%workgroup_id_2, "),
        # Add workgroup_id_2
        (
            "%workgroup_id_1 = stream.dispatch.workgroup.id[1] : index",
            "%workgroup_id_1 = stream.dispatch.workgroup.id[1] : index\n        %workgroup_id_2 = stream.dispatch.workgroup.id[2] : index",
        ),
        # replace grid size for batch dim
        (
            "%c1 = arith.constant 1 : index\n      stream.return",
            f"%c1 = arith.constant {B} : index\n      stream.return",
        ),
        # C type
        (
            f"memref<{M}x{N}x{output_dtype}, strided<[{N}, 1]",
            f"memref<{B}x{M}x{N}x{output_dtype}, strided<[{M*N}, {N}, 1]",
        ),
        # writing to C
        (f"%{C_ssa_val}[", f"%{C_ssa_val}[%workgroup_id_2, "),
        # A tensor input
        (f"tensor<{M}x{K}x{input_dtype}>", f"tensor<{B}x{M}x{K}x{input_dtype}>"),
        # C tensor output
        (f"tensor<{M}x{N}x{output_dtype}>", f"tensor<{B}x{M}x{N}x{output_dtype}>"),
    ]

    with open(file_name, "r") as file:
        data = file.read()

        for search_text, replace_text in replacements:
            data = replace_in_file(data, search_text, replace_text)

        with open(f"bmma.mlir", "w") as file:
            file.write(data)
