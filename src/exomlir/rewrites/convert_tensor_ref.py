from xdsl.context import Context
from xdsl.dialects import arith, memref
from xdsl.dialects.builtin import (
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
    NoneAttr,
    StridedLayoutAttr,
    f16,
    f32,
    f64,
    f80,
    f128,
)
from xdsl.dialects.utils import get_dynamic_index_list
from xdsl.ir import Operation, OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from exomlir.dialects import exo


class ConvertReadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.ReadOp, rewriter: PatternRewriter):
        # convert tensor reads only
        if len(op.indices) < 1:
            return

        rewriter.replace_matched_op(memref.LoadOp.get(op.input, op.indices))


class ConvertAssignOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.AssignOp, rewriter: PatternRewriter):
        # convert tensor writes only
        if len(op.indices) < 1:
            return

        assert isinstance(op.input.type, MemRefType)

        # if the value is a scalar memref, we need to load
        if isinstance(op.value.type, MemRefType):
            assert op.value.type.get_shape() == (1,), (
                f"Expected scalar memref type, got {op.value.type}"
            )

            return rewriter.replace_matched_op(
                (
                    zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                    load_op := memref.LoadOp.get(op.value, [zero_op.result]),
                    memref.StoreOp.get(load_op.res, op.input, op.indices),
                )
            )

        rewriter.replace_matched_op(memref.StoreOp.get(op.value, op.input, op.indices))


class ConvertReduceOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.ReduceOp, rewriter: PatternRewriter):
        # convert tensor writes only
        if len(op.indices) < 1:
            return

        assert isinstance(op.input.type, MemRefType)

        ops = []
        value = op.value

        # if the value is a scalar memref, we need to load
        if isinstance(op.value.type, MemRefType):
            assert op.value.type.get_shape() == (1,), (
                f"Expected scalar memref type, got {op.value.type}"
            )

            ops.append(zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())))
            ops.append(
                load_op := memref.LoadOp.get(op.value, [zero_op.result]),
            )
            value = load_op.res

        load_op = memref.LoadOp.get(op.input, op.indices)

        # switch on value type
        if value.type in [f16, f32, f64, f80, f128]:
            add_op = arith.AddfOp(
                operand1=load_op.results[0],
                operand2=value,
                flags=arith.FastMathFlagsAttr("none"),
                result_type=op.input.type.element_type,
            )

        else:
            add_op = arith.AddiOp(
                operand1=load_op.results[0],
                operand2=value,
                result_type=op.input.type.element_type,
            )

        rewriter.replace_matched_op(
            (
                *ops,
                load_op,
                add_op,
                memref.StoreOp.get(add_op.result, op.input, op.indices),
            ),
        )


class ConvertWindowOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.WindowOp, rewriter: PatternRewriter):
        input_sizes = op.static_input_sizes.get_values()
        output_sizes = op.static_output_sizes.get_values()

        # compute sizes
        sizes = [1] * (len(input_sizes) - len(output_sizes)) + get_dynamic_index_list(
            op.static_output_sizes.get_values(),
            op.output_sizes,
            memref.SubviewOp.DYNAMIC_INDEX,
        )

        # compute indices
        indices = []
        for operand in op.indices:
            if isinstance(operand, OpResult) and isinstance(operand.op, exo.IntervalOp):
                indices.append(operand.op.start)
            else:
                indices.append(operand)

        # compute strides - these should be the same as the input strides, which we calculate using
        # strides[0] = 1
        # strides[i] = strides[i - 1] * sizes[i]
        strides = []
        dynamic_stride = False
        dynamic_idx = 0
        stride = 1
        ops: list[Operation] = []

        for dim in reversed(input_sizes):
            strides.insert(0, stride)

            if dynamic_stride:
                if dim == memref.SubviewOp.DYNAMIC_INDEX:
                    # if we encounter another dynamic index, we need to multiply the stride by the
                    # dynamic size
                    ops.append(
                        arith.MuliOp(
                            operand1=ops[-1].results[0],
                            operand2=op.input_sizes[dynamic_idx],
                        )
                    )
                    dynamic_idx += 1
                else:
                    # otherwise we can multiply the stride by the static size
                    ops.append(
                        const_op := arith.ConstantOp(IntegerAttr(dim, IndexType())),
                    )
                    ops.append(
                        arith.MuliOp(
                            operand1=ops[-1].results[0],
                            operand2=const_op.result,
                        ),
                    )

                continue

            # when we encounter the first dynamic index, we need to start using the dynamic stride
            if dim == memref.SubviewOp.DYNAMIC_INDEX:
                dynamic_stride = True
                ops.append(arith.ConstantOp(IntegerAttr(stride, IndexType())))
            # otherwise, we can continue to compute the stride statically
            else:
                stride *= dim

        rewriter.replace_matched_op(
            (
                memref.SubviewOp.get(
                    op.input,
                    indices,
                    sizes,
                    strides,
                    MemRefType(
                        op.result.type.element_type,
                        op.result.type.shape,
                        StridedLayoutAttr(
                            strides[len(input_sizes) - len(output_sizes) :], NoneAttr()
                        ),
                        op.result.type.memory_space,
                    ),
                ),
            )
        )


class EraseUnusedIntervalOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.IntervalOp, rewriter: PatternRewriter):
        # erase unused interval ops
        if len(op.result.uses) == 0:
            rewriter.erase_matched_op(op)


class ConvertTensorRefPass(ModulePass):
    name = "convert-tensor-ref"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertReadOp(),
                    ConvertReduceOp(),
                    ConvertAssignOp(),
                    ConvertWindowOp(),
                    EraseUnusedIntervalOp(),
                ]
            )
        ).rewrite_module(m)
