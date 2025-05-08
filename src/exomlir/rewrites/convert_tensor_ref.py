from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, IndexType, MemRefType, IntegerAttr
from xdsl.dialects import arith, memref
from xdsl.passes import ModulePass
from xdsl.ir import OpResult

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
                    load_op.results[0], op.value, op.input.type.element_type
                ),
                memref.StoreOp.get(add_op.result, op.input, op.indices),
            ),
        )


class ConvertWindowOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.WindowOp, rewriter: PatternRewriter):
        ops = [
            zero := arith.ConstantOp(IntegerAttr(0, IndexType())),
            one := arith.ConstantOp(IntegerAttr(1, IndexType())),
        ]

        offsets = []
        sizes = []
        strides = []
        static_offsets = []
        static_sizes = []
        static_strides = []
        shape = []

        for operand in op.indices:
            #  if the operand is an interval, compute size
            if isinstance(operand, OpResult) and isinstance(operand.op, exo.IntervalOp):
                ops.append(
                    sub_op := arith.SubiOp(
                        operand.op.end, operand.op.start, IndexType()
                    )
                )
                offsets.append(operand.op.start)
                sizes.append(sub_op.result)
                strides.append(one.result)
                static_offsets.append(memref.SubviewOp.DYNAMIC_INDEX)
                static_sizes.append(memref.SubviewOp.DYNAMIC_INDEX)
                static_strides.append(1)
                shape.append(-1)
            else:
                offsets.append(operand)
                sizes.append(one.result)
                strides.append(one.result)
                static_offsets.append(0)
                static_sizes.append(1)
                static_strides.append(1)
                shape.append(1)

        output_type = MemRefType(
            op.result.type.element_type,
            shape,
            op.result.type.layout,
            op.result.type.memory_space,
        )

        output_dims = len(op.result.type.shape.data)
        output_shape_ops = []
        for s in op.result.type.get_shape():
            if s == -1:
                output_shape_ops.append(zero.result)
            else:
                output_shape_ops.append(s)

        rewriter.replace_matched_op(
            (
                *ops,
                subview_op := memref.SubviewOp(
                    source=op.input,
                    offsets=offsets,
                    sizes=sizes,
                    strides=strides,
                    static_offsets=static_offsets,
                    static_sizes=static_sizes,
                    static_strides=static_strides,
                    result_type=output_type,
                ),
                memref.ReinterpretCastOp(
                    subview_op.result,
                    [],
                    [],
                    [],
                    [0] * output_dims,
                    op.result.type.get_shape(),
                    [1] * output_dims,
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
