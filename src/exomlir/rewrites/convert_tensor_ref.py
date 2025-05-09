from typing import cast
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, IndexType, MemRefType, IntegerAttr
from xdsl.dialects import arith, memref
from xdsl.dialects.utils import get_dynamic_index_list
from xdsl.passes import ModulePass
from xdsl.ir import Operation, OpResult

from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
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

        rewriter.replace_matched_op(memref.StoreOp.get(op.value, op.input, op.indices))


class ConvertReduceOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.ReduceOp, rewriter: PatternRewriter):
        # convert tensor writes only
        if len(op.indices) < 1:
            return

        assert isinstance(op.input.type, MemRefType)

        rewriter.replace_matched_op(
            (
                load_op := memref.LoadOp.get(op.input, op.indices),
                add_op := arith.AddfOp(
                    operand1=load_op.results[0],
                    operand2=op.value,
                    flags=arith.FastMathFlagsAttr("none"),
                    result_type=op.input.type.element_type,
                ),
                memref.StoreOp.get(add_op.result, op.input, op.indices),
            ),
        )


class ConvertWindowOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.WindowOp, rewriter: PatternRewriter):
        input_shape = cast(MemRefType, op.input.type).get_shape()
        output_shape = cast(MemRefType, op.result.type).get_shape()

        sizes = get_dynamic_index_list(
            op.static_sizes.get_values(),
            op.input_sizes,
            memref.SubviewOp.DYNAMIC_INDEX,
        )[: len(input_shape) - len(output_shape)] + list(output_shape)

        # compute indices
        indices = []
        for operand in op.indices:
            if isinstance(operand, OpResult) and isinstance(operand.op, exo.IntervalOp):
                indices.append(operand.op.start)
            else:
                indices.append(operand)

        # compute strides
        strides = []
        dynamic_stride = False
        dynamic_idx = 0
        stride = 1
        ops: list[Operation] = []

        for dim in reversed(input_shape):
            if dynamic_stride:
                ops.append(
                    mul_op := arith.MuliOp(
                        operand1=ops[-1].results[0],
                        operand2=op.input_sizes[dynamic_idx],
                    )
                )
                dynamic_idx += 1
                strides.insert(0, mul_op.results[0])
                continue

            if dim == -1:
                dynamic_stride = True
                ops.append(arith.ConstantOp(IntegerAttr(stride, IndexType())))

            strides.insert(0, stride)
            stride *= dim

        rewriter.replace_matched_op(
            (
                subview_op := memref.SubviewOp.get(
                    op.input, indices, sizes, strides, op.result.type
                ),
                memref.CastOp.get(
                    subview_op.result,
                    op.result.type,
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
