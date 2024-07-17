from typing import Any, Callable, Type, Optional, Sequence, Union, List
from dataclasses import dataclass
import sympy
import torch.fx as fx
import torch.utils._pytree as pytree

from .._support.indexing import (
    IndexExpr,
    IndexingContext,
    IndexSymbol,
    SymIndex,
    index_expr,
)

from .._support.tracing import CapturedTrace
from .ops import (
    alloc_shared,
    barrier,
    construct_register_from_metadata,
    get_result,
    mma,
    read,
    read_shared,
    tiled_loop,
    write,
    write_shared,
    sched_barrier,
    sched_group_barrier,
)
from ..compiler.builder import (
    IRProxyValue,
    ScalarBuilder,
)

from ..compiler.base import (
    CodegenError,
    NDEBUG,
    ValidationError,
)

from ..compiler.ir import (
    AffineMap,
    Attribute,
    AffineExpr,
    AffineMapAttr,
    ArrayAttr,
    FunctionType,
    VectorType,
    DenseElementsAttr,
    F16Type,
    F32Type,
    IndexType,
    FloatAttr,
    InsertionPoint,
    IntegerType,
    IntegerAttr,
    IrType,
    Location,
    MemRefType,
    OpResult,
    ShapedType,
    Value,
    VectorType,
    amdgpu_d,
    arith_d,
    func_d,
    gpu_d,
    llvm_d,
    math_d,
    memref_d,
    vector_d,
    scf_d,
    stream_d,
)

__all__ = ["handle_read", "handle_write"]

from .. import lang as tkl

from ..compiler.kernel_codegen import (
    BoundKernelSignature,
)

from ..compiler.vector_codegen import (
    cast_py_literal,
    cast_py_value,
    cast_kernel_buffer,
    cast_slice_spec,
    cast_vector,
    extract_slice_starts,
)
import operator as py_operator


@dataclass
class NodeAttrs:
    # By default, integers are assumed signed. We propagate unsigned as graph
    # node attrs.
    unsigned: bool = False

    @staticmethod
    def load(py_value) -> "NodeAttrs":
        if isinstance(py_value, fx.Node):
            return NodeAttrs(unsigned=bool(py_value.meta.get("unsigned")))
        return NodeAttrs()

    def store(self, node: fx.Node):
        node.meta["unsigned"] = self.unsigned


class WaveEmitter:
    """Emits a 'warp function' as a `func` with a signature derived from the gm."""

    OP_HANDLERS: dict[Any, Callable[["WaveEmitter", fx.Node], None]] = {}

    def __init__(self, root_sig: BoundKernelSignature, trace: CapturedTrace):
        self._node_values: dict[fx.Node, List[IRProxyValue]] = {}
        self._root_sig = root_sig
        self.trace = trace
        self.induction_var: Optional[Value] = None
        self.ip = InsertionPoint(root_sig.entry_block)
        self.thread_ids = []
        self.workgroup_ids = []

    def lookup_node_values(self, node: fx.Node) -> List[Value]:
        assert NDEBUG or isinstance(node, fx.Node)
        values = self._node_values.get(node)
        if values is None:
            values = [self._root_sig.resolve_by_reference(("node", node))]
            self._node_values[node] = values
        return values

    def bind_node_proxy(
        self, node: fx.Node, proxy: IRProxyValue, *, attrs: Optional[NodeAttrs] = None
    ):
        """Binds a node's result to a Python/IR proxy object."""
        assert NDEBUG or (isinstance(node, fx.Node) and isinstance(proxy, IRProxyValue))
        assert (
            node not in self._node_values
        ), f"Cannot rebind node {node}: already bound"
        if attrs is not None:
            attrs.store(node)
        self._node_values[node] = [proxy]

    def bind_node_proxies(
        self,
        node: fx.Node,
        proxies: list[IRProxyValue],
        *,
        attrs: Optional[NodeAttrs] = None,
    ):
        """Binds a node's result to a list of Python/IR proxy object."""
        assert NDEBUG or (
            all(isinstance(p, IRProxyValue) for p in proxies)
            and isinstance(node, fx.Node)
        )
        assert (
            node not in self._node_values
        ), f"Cannot rebind node {node}: already bound"
        if attrs is not None:
            attrs.store(node)
        self._node_values[node] = proxies

    def emit_program_invariants(self):
        self.workgroup_ids = [
            stream_d.dispatch_workgroup_id(IntegerAttr.get(IndexType.get(), 0)),
            stream_d.dispatch_workgroup_id(IntegerAttr.get(IndexType.get(), 1)),
            stream_d.dispatch_workgroup_id(IntegerAttr.get(IndexType.get(), 2)),
        ]
        self.thread_ids = [
            gpu_d.thread_id(gpu_d.Dimension.x),
            gpu_d.thread_id(gpu_d.Dimension.y),
            gpu_d.thread_id(gpu_d.Dimension.z),
        ]

    def emit(self, graph: fx.Graph = None):
        with self.ip, Location.unknown():
            self.emit_program_invariants()
            self.emit_graph(graph if graph is not None else self.trace.get_root_graph())

    def emit_function_call_node(self, node: fx.Node):
        target_op = node.target
        try:
            handler = self.OP_HANDLERS[target_op]
        except KeyError:
            raise CodegenError(f"No handler registered for op {target_op}")
        handler(self, node)
        # dump

    def emit_graph(self, graph: fx.Graph):
        """Emits the given graph at the current insertion point."""
        for node in graph.nodes:
            if node.op == "call_function" or node.op == "call_method":
                self.emit_function_call_node(node)
            if node.op == "output":
                return node.args

    def emit_subgraph(
        self,
        subgraph: fx.Graph,
        implicit_capture: list[fx.Node],
        freevars: list[fx.Node] = None,
    ):
        if freevars is None:
            # Map subgraph freevars -> implicit_capture
            freevars = self.trace.region_graph.inner_freevars[subgraph]
            assert len(freevars) == len(
                implicit_capture
            ), f"Expected {len(freevars)} implicit capture args, got {len(implicit_capture)}"
            for freevar, arg in zip(freevars, implicit_capture):
                self._node_values[freevar.node] = self.lookup_node_values(arg)

        # Emit subgraph
        return self.emit_graph(subgraph)

    def finish(self):
        with self.ip, Location.unknown():
            func_d.ReturnOp([])


def handle_op(op):
    def decorator(
        f: Callable[[WaveEmitter, fx.Node], None]
    ) -> Callable[[WaveEmitter, fx.Node], None]:
        WaveEmitter.OP_HANDLERS[op] = f
        return f

    return decorator


###############################################################################
# Python/scalar ops
###############################################################################


###############################################################################
# Memory Ops
###############################################################################
@handle_op(alloc_shared)
def handle_alloc_shared(emitter: WaveEmitter, node: fx.Node):
    try:
        shape, dtype, type = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    # Add padding to avoid bank conflicts.
    new_shape = [shape[0], shape[1] + 4]
    if "num_buffers" in node.meta and node.meta["num_buffers"] > 1:
        new_shape = [node.meta["num_buffers"]] + new_shape
    memref_shape = cast_py_literal(emitter, new_shape)
    element_type = IrType.parse(dtype.ir_type_asm())
    address_space = Attribute.parse("#gpu.address_space<workgroup>")
    memref_type = MemRefType.get(memref_shape, element_type, None, address_space)
    alloc = memref_d.alloc(memref_type, [], [])
    emitter.bind_node_proxy(node, IRProxyValue(alloc))


@handle_op(barrier)
def handle_barrier(emitter: WaveEmitter, node: fx.Node):
    amdgpu_d.lds_barrier()


@handle_op(construct_register_from_metadata)
def handle_construct_register_from_metadata(emitter: WaveEmitter, node: fx.Node):
    try:
        shape, dtype, value = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    # TODO: This vector shape needs to be propagated through the graph.
    # For now, just hardcoding to get MLIR emission working again.
    vector_shape = cast_py_literal(emitter, (emitter.acc_vector_shape,))
    element_type = IrType.parse(dtype.ir_type_asm())
    register = ScalarBuilder.constant_vector(value, vector_shape, element_type)
    emitter.bind_node_proxy(node, register)


def gen_sympy_index(emitter: WaveEmitter, expr: sympy.Expr, stage: int) -> OpResult:
    stack: list[OpResult] = []
    # TODO: factor this out
    all_symbols = emitter.thread_ids + emitter.workgroup_ids + [emitter.induction_var]
    dynamics = dict(zip(["TX", "TY", "TZ", "WG0", "WG1", "WG2", "ARG0"], all_symbols))
    idxc = IndexingContext.current()
    # Substitute in frozen vars to simplify expression.
    if not isinstance(expr, sympy.Expr):
        expr = sympy.sympify(expr)
    expr = expr.subs(idxc.subs)
    # Why affine, for now simply create indexing expressions.
    # This can easily be adapted to affine expressions later.
    for term in sympy.postorder_traversal(expr):
        match term:
            case sympy.Symbol():
                if term in idxc.subs.keys():
                    cst = arith_d.constant(IndexType.get(), idxc.subs[term])
                    stack.append(cst)
                elif term.name in dynamics.keys():
                    if dynamics[term.name] is None and term.name == "ARG0":
                        print(
                            f"induction var is accessed outside of the loop. Setting to stage {stage}"
                        )
                        stack.append(arith_d.constant(IndexType.get(), stage))
                    else:
                        stack.append(dynamics[term.name])
                else:
                    raise CodegenError(f"Unknown symbol {term}")
            case sympy.Integer():
                stack.append(arith_d.constant(IndexType.get(), int(term)))
            case sympy.Mul():
                args = []
                for _ in range(len(term.args)):
                    args.append(stack.pop())
                operation = None
                # First, multiply all the non-rationals.
                for arg in args:
                    if callable(arg):
                        continue
                    if operation is None:
                        operation = arg
                        continue
                    operation = arith_d.MulIOp(operation, arg)
                # Then, multiply with the rationals.
                for arg in args:
                    if callable(arg):
                        operation = arg(operation)
                stack.append(operation)
            case sympy.Add():
                summand = stack.pop()
                add = summand
                for _ in range(1, len(term.args)):
                    add = arith_d.AddIOp(add, stack.pop())
                stack.append(add)
            case sympy.Mod():
                rhs = stack.pop()
                lhs = stack.pop()
                mod = arith_d.RemSIOp(lhs, rhs)
                stack.append(mod)
            case sympy.floor():
                # TODO: Since divsi rounds to zero, this seems to work.
                # But check whether floordivsi is needed.
                stack.append(stack.pop())
            case sympy.Rational():
                numerator = arith_d.constant(IndexType.get(), abs(term.p))
                denominator = arith_d.constant(IndexType.get(), abs(term.q))
                # Assumes that the negative term is always carried on the numerator
                if abs(term.p) > term.p:
                    zero = arith_d.constant(IndexType.get(), int(0))
                    numerator = arith_d.SubIOp(zero, numerator)
                mul = lambda x: x
                if abs(term.p) != 1:
                    mul = lambda x: arith_d.MulIOp(x, numerator)
                operation = lambda x: arith_d.DivSIOp(mul(x), denominator)
                stack.append(operation)
            case sympy.UnevaluatedExpr():
                continue
            case _:
                raise CodegenError(f"Can not handle {term} yet")
    if len(stack) != 1:
        raise CodegenError(f"Expected single result, got {len(stack)}")
    return stack[0]


@handle_op(read_shared)
@handle_op(read)
def handle_read(emitter: WaveEmitter, node: fx.Node):
    # This is similar to tkl.store with fixed start indices for now.
    try:
        memory, elements_per_thread, _ = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    vector_shape = cast_py_literal(emitter, (elements_per_thread,))
    # memory has no IR node yet.
    kb_src, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, memory)

    stage = 0
    if "stage" in node.meta:
        stage = node.meta["stage"]
    start_indices = []
    if isinstance(node.meta["index"], sympy.Add):
        pass
    for dim_indexing in node.meta["index"]:
        start_indices.append(gen_sympy_index(emitter, dim_indexing, stage))

    element_type = kb_ir_type.element_type
    vector_type = VectorType.get(vector_shape, element_type)
    result = vector_d.load(vector_type, kb_src, start_indices)
    emitter.bind_node_proxy(node, IRProxyValue(result))


@handle_op(write_shared)
@handle_op(write)
def handle_write(emitter: WaveEmitter, node: fx.Node):
    try:
        register, memory, elements_per_thread = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    kb_dest, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, memory)
    dest_rank = kb_ir_type.rank
    ref_shape = kb_py_type.symbolic_shape
    # slice_spec = cast_slice_spec(emitter, ref_shape, multi_index)
    # start_indices = extract_slice_starts(emitter, ref_shape, slice_spec)

    stage = 0
    if "stage" in node.meta:
        stage = node.meta["stage"]
    start_indices = []
    for dim_indexing in node.meta["index"]:
        start_indices.append(gen_sympy_index(emitter, dim_indexing, stage))

    if dest_rank != len(start_indices):
        raise CodegenError(
            f"Mismatched slice assignment: Expected rank {dest_rank}, got {len(start_indices)}"
        )
    # TODO: This fails currently because the register is not properly resolved.
    #       It stems from the function call.
    idxc = IndexingContext.current()
    store_elements = elements_per_thread.subs(idxc.subs)

    insert_vector = cast_vector(emitter, register, element_type=kb_ir_type.element_type)
    insert_type = VectorType(insert_vector.type)
    insert_rank = insert_type.rank

    # Special case rank-0 broadcast.
    if insert_rank == 0:
        broadcast_type = VectorType.get(dest_rank * [1], kb_ir_type.element_type)
        insert_vector = vector_d.broadcast(broadcast_type, insert_vector)

    # If vector length > store elements, we need to extract one at a time and then store
    # Assumes that vector shape will always be 1D, which may not always be true?
    vector_shape = insert_type.shape[0]
    if vector_shape > store_elements and store_elements == 1:
        result_type = VectorType.get([1], kb_ir_type.element_type)
        for i in range(vector_shape):
            index = arith_d.constant(
                IndexType.get(), int((i % 4) + emitter.offset_fn(i))
            )
            element = vector_d.extract_strided_slice(
                result_type,
                insert_vector,
                [i],
                [1],
                [1],
            )
            # TODO: This is hardcoded to match mma layout and should be deduced automatically
            modified_indices = [index for index in start_indices]
            modified_indices[0] = arith_d.addi(start_indices[0], index)
            vector_d.store(element, kb_dest, modified_indices)
        return

    vector_d.store(insert_vector, kb_dest, start_indices)


###############################################################################
# Math Ops
###############################################################################
@handle_op(mma)
def handle_mma(emitter: WaveEmitter, node: fx.Node):
    # TODO: lhs, rhs, acc are actually registers, not vectors.
    #       Currently this is handled exactly like tkl.dot
    try:
        lhs, rhs, acc = node.args
        acc = cast_vector(emitter, acc)

        values = [lhs, rhs]
        if "extract" in node.meta:
            for index, offset, size in node.meta["extract"]:
                ir_value = cast_py_value(emitter, values[index]).ir_value
                element_type = ir_value.type.element_type
                shape = ir_value.type.shape
                vector_type = VectorType.get([size], element_type)
                values[index] = vector_d.extract_strided_slice(
                    vector_type, ir_value, [offset], [size], [1]
                )
        else:
            for i in range(len(values)):
                values[i] = cast_vector(emitter, values[i])

    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    vector_type = VectorType(acc.type)
    element_type = vector_type.element_type
    rank = vector_type.rank
    mma_m, mma_n, mma_k = emitter.mma_matrix_shapes

    m = ScalarBuilder.constant_attr(mma_m, IntegerType.get_signless(32))
    n = ScalarBuilder.constant_attr(mma_n, IntegerType.get_signless(32))
    k = ScalarBuilder.constant_attr(mma_k, IntegerType.get_signless(32))
    blocks = ScalarBuilder.constant_attr(1, IntegerType.get_signless(32))

    result = amdgpu_d.mfma(
        dest_d=vector_type,
        m=m,
        n=n,
        k=k,
        blocks=blocks,
        source_a=values[0],
        source_b=values[1],
        dest_c=acc,
    )

    emitter.bind_node_proxy(node, IRProxyValue(result))


@handle_op(py_operator.add)
def handle_add(emitter: WaveEmitter, node: fx.Node):
    try:
        lhs, rhs = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    lhs = cast_py_value(emitter, lhs)
    rhs = cast_py_value(emitter, rhs)

    def get_type(value: IRProxyValue):
        return value.ir_value.type

    # TODO: this is too hard coded
    if (lhs_type := get_type(lhs)) != (rhs_type := get_type(rhs)):
        if isinstance(lhs_type.element_type, F32Type):
            lhs = arith_d.truncf(
                VectorType.get(lhs_type.shape, F16Type.get()), lhs.ir_value
            )
            rhs = rhs.ir_value
        elif isinstance(rhs_type.element_type, F32Type):
            lhs = lhs.ir_value
            rhs = arith_d.truncf(
                VectorType.get(rhs_type.shape, F16Type.get()), rhs.ir_value
            )
    else:
        lhs = lhs.ir_value
        rhs = rhs.ir_value

    result = arith_d.addf(lhs, rhs)
    emitter.bind_node_proxy(node, IRProxyValue(result))


###############################################################################
# Control Flow ops
###############################################################################


@handle_op(tiled_loop)
def handle_tiled_loop(emitter: WaveEmitter, node: fx.Node):
    # Note: Adapted from tk.for_loop
    try:
        axis, init_args, subgraph, implicit_capture = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    # Check if init_args is a flattened list of values.
    for arg in init_args:
        if len(emitter.lookup_node_values(arg)) != 1:
            raise CodegenError(f"NYI: For loop init args must be flattened")

    # Get IR values mapping to the node args.
    # TODO: Hardcoded sizes should be dynamic
    start = arith_d.constant(IndexType.get(), int(node.meta["start"]))
    end = arith_d.constant(IndexType.get(), int(node.meta["end"]))
    step = arith_d.constant(IndexType.get(), int(node.meta["step"]))

    # Flatten init_args and get IR values for each of them.
    flat_init_args, init_args_spec = pytree.tree_flatten((init_args))
    flat_init_args = [cast_py_value(emitter, arg) for arg in flat_init_args]

    if "subgraph" in node.kwargs:
        subgraph = node.kwargs["subgraph"]
    else:
        # Get the subgraph for body of the loop.
        assert isinstance(subgraph, str)
        subgraph = emitter.trace.get_subgraph(subgraph)

    # Create scf.for operation.
    forOp = scf_d.ForOp(
        start,
        end,
        step,
        [a.ir_value for a in flat_init_args],
    )
    # Enter body of for loop.
    with InsertionPoint(forOp.body):
        # TODO: Flatten subgraph args here.
        if "iter_args" in node.kwargs:
            subgraph_args = node.kwargs["iter_args"]
        else:
            subgraph_args = [
                node
                for node in subgraph.nodes
                if node.op == "placeholder" and "lifted" not in node.meta
            ]

        # Add mapping for iter_args.
        for i, v in enumerate(forOp.inner_iter_args):
            emitter.bind_node_proxy(subgraph_args[i], IRProxyValue(v))

        emitter.induction_var = forOp.induction_variable
        ret = emitter.emit_subgraph(subgraph, implicit_capture, implicit_capture)
        # Use ret in terminatory of body
        # TODO: Flatten return values here.
        flat_ret_values, ret_spec = pytree.tree_flatten((ret))
        flat_ret_values = [
            cast_py_value(emitter, value).ir_value for value in flat_ret_values
        ]
        scf_d.YieldOp(flat_ret_values)

    # Load and Store ops emitted after the loop will refer to this instead of
    # the induction var.
    # TODO: Should be dynamic and depend on the loop trip count
    emitter.induction_var = arith_d.constant(IndexType.get(), node.meta["end"])
    results = forOp.results_
    # TODO: All results are bound to this node so we lose context here.
    emitter.bind_node_proxies(node, [IRProxyValue(v) for v in results])


@handle_op(get_result)
def handle_get_result(emitter: WaveEmitter, node: fx.Node):
    try:
        value, index = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    for_op = emitter.lookup_node_values(value)[0].ir_value.owner
    emitter.bind_node_proxy(node, IRProxyValue(for_op.results[index]))


@handle_op(sched_barrier)
def handle_sched_barrier(emitter: WaveEmitter, node: fx.Node):
    try:
        mask = node.args[0]
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    mask = arith_d.constant(IntegerType.get_signless(32), mask)
    llvm_d.call_intrinsic(None, "llvm.amdgcn.sched.barrier", [mask])


def get_mask(name: str) -> int:
    match name:
        case "read":
            return int("0x20", 0)
        case "write":
            return int("0x40", 0)
        case "read_shared":
            return int("0x100", 0)
        case "write_shared":
            return int("0x200", 0)
        case "mma":
            return int("0x8", 0)
    return None


@handle_op(sched_group_barrier)
def handle_sched_group_barrier(emitter: WaveEmitter, node: fx.Node):
    try:
        instruction_counts, sync_id = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    sync_id = arith_d.constant(IntegerType.get_signless(32), sync_id)
    for name, number in instruction_counts.items():
        if get_mask(name) is None:
            continue
        mask = arith_d.constant(IntegerType.get_signless(32), get_mask(name))
        number = arith_d.constant(IntegerType.get_signless(32), number)
        llvm_d.call_intrinsic(
            None, "llvm.amdgcn.sched.group.barrier", [mask, number, sync_id]
        )


###############################################################################
# Shape Manipulation Ops
###############################################################################

###############################################################################
# Conversion utilities
###############################################################################

###############################################################################
# Slice and indexing
###############################################################################
