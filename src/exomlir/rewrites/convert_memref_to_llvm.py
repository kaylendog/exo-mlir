from xdsl.context import Context
from xdsl.dialects import arith, llvm, memref
from xdsl.dialects.builtin import (
    IntegerAttr,
    ModuleOp,
    UnrealizedConversionCastOp,
    MemRefType,
    i8,
    i16,
    i32,
    i64,
)
from xdsl.dialects.utils import get_dynamic_index_list
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    TypeConversionPattern,
    op_type_rewrite_pattern,
    attr_type_rewrite_pattern,
)

from exomlir.dialects import exo


def compute_memref_strides(
    sizes: list[SSAValue[Attribute] | int],
) -> tuple[list[SSAValue[Attribute]], list[Operation]]:
    ops = [arith.ConstantOp(IntegerAttr(1, i64))]
    strides = [ops[0].result]

    # strides are built in reverse order, and we do not care about the last
    for size in sizes[:-1]:
        if isinstance(size, int):
            ops.append(
                const_op := arith.ConstantOp(IntegerAttr(size, i64)),
            )
            next_size = const_op.result
        else:
            next_size = size

        # next stride is the product of the current stride and the next size
        ops.append(
            mul_op := arith.MulIOp(
                ops[-1].result,
                next_size,
            ),
        )

        strides.append(mul_op.result)

    return (
        strides[::-1],
        ops,
    )


def compute_offsets(
    indices: list[SSAValue[Attribute]],
    sizes: list[SSAValue[Attribute] | int],
) -> tuple[list[SSAValue[Attribute]], list[Operation]]:
    """
    Compute the offsets of the given memref type.
    """
    strides, ops = compute_memref_strides(sizes)
    offsets = []

    for idx, stride in zip(indices, strides):
        ops.append(
            mul_op := arith.MulIOp(
                idx,
                stride,
            ),
        )
        offsets.append(mul_op.result)

    return (
        offsets,
        ops,
    )


class ConvertReadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.ReadOp, rewriter: PatternRewriter):
        offsets, ops = compute_offsets(
            op.indices, get_dynamic_index_list(op.static_sizes, op.sizes, -1)
        )
        rewriter.replace_matched_op(
            (
                *ops,
                cast_op := UnrealizedConversionCastOp.get(
                    [op.input], [llvm.LLVMPointerType.opaque()]
                ),
                # get pointer and load
                gep_op := llvm.GEPOp(
                    cast_op.results[0],
                    [llvm.GEP_USE_SSA_VAL] * len(offsets),
                    offsets,
                    op.result.type,
                ),
                llvm.LoadOp(gep_op),
            )
        )


class ConvertAssignOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.AssignOp, rewriter: PatternRewriter):
        offsets, ops = compute_offsets(
            op.indices, get_dynamic_index_list(op.static_sizes, op.sizes, -1)
        )
        rewriter.replace_matched_op(
            (
                *ops,
                cast_op := UnrealizedConversionCastOp.get(
                    [op.input], [llvm.LLVMPointerType.opaque()]
                ),
                gep_op := llvm.GEPOp(
                    cast_op.results[0],
                    [llvm.GEP_USE_SSA_VAL] * len(offsets),
                    offsets,
                    op.value.type,
                ),
                llvm.StoreOp(
                    op.value,
                    gep_op,
                ),
            )
        )


class ConvertReduceOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.ReduceOp, rewriter: PatternRewriter):
        offsets, ops = compute_offsets(
            op.indices, get_dynamic_index_list(op.static_sizes, op.sizes, -1)
        )

        cast_op = UnrealizedConversionCastOp.get(
            [op.input], [llvm.LLVMPointerType.opaque()]
        )

        # get pointer and load
        gep_op = llvm.GEPOp(
            cast_op.results[0],
            [llvm.GEP_USE_SSA_VAL] * len(offsets),
            offsets,
            op.value.type,
        )
        load_op = llvm.LoadOp(gep_op, op.value.type)

        # reduce
        if op.value.type in [i8, i16, i32, i64]:
            add_op = arith.AddiOp(
                load_op.dereferenced_value,
                op.value,
            )
        else:
            add_op = arith.AddfOp(
                load_op.dereferenced_value,
                op.value,
            )

        rewriter.replace_matched_op(
            (
                *ops,
                cast_op,
                gep_op,
                load_op,
                add_op,
                llvm.StoreOp(
                    add_op.result,
                    gep_op,
                ),
            )
        )


class ConvertWindowOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.WindowOp, rewriter: PatternRewriter):
        offsets, ops = compute_offsets(
            op.indices,
            get_dynamic_index_list(
                op.static_input_sizes, op.sizes, memref.SubviewOp.DYNAMIC_INDEX
            ),
        )

        rewriter.replace_matched_op(
            (
                *ops,
                cast_op := UnrealizedConversionCastOp.get(
                    [op.input], [llvm.LLVMPointerType.opaque()]
                ),
                gep_op := llvm.GEPOp(
                    cast_op.results[0],
                    [llvm.GEP_USE_SSA_VAL] * len(offsets),
                    offsets,
                    op.result.type.element_type,
                ),
                UnrealizedConversionCastOp.get(
                    [gep_op.result], [op.result.type.element_type, op.result.type]
                ),
            )
        )


class RewriteMemRefTypes(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, type: MemRefType):
        return llvm.LLVMPointerType.opaque()


class ConvertMemRefToLLVM(ModulePass):
    name = "convert-memref-to-llvm"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertReadOp(),
                    ConvertAssignOp(),
                    ConvertReduceOp(),
                    ConvertWindowOp(),
                    RewriteMemRefTypes(),
                ]
            ),
        ).rewrite_module(m)
