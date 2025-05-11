from xdsl.context import Context
from xdsl.dialects import arith, func, scf
from xdsl.dialects.builtin import (
    FunctionType,
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
    NoneAttr,
)
from xdsl.ir import BlockArgument, Region, Use
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from exomlir.dialects import exo, index


class ConvertAllocOpToScalarOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.AllocOp, rewriter: PatternRewriter):
        # we cannot convert tensors
        if isinstance(op.result.type, MemRefType):
            return

        # walk to find the next use
        target = op.next_op
        while target is not None:
            # bail if we see control flow
            if isinstance(target, scf.ConditionOp) or isinstance(target, scf.ForOp):
                return

            if not isinstance(target, exo.AssignOp) or not isinstance(
                target.value.owner, arith.ConstantOp
            ):
                target = target.next_op
                continue

            rewriter.replace_matched_op(
                exo.ScalarOp(target.value.owner.value, op.mem.data, op.result.type)
            )
            rewriter.erase_op(target)
            return


class ConvertRedundantReads(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.ReadOp, rewriter: PatternRewriter):
        # convert scalar reads only
        if isinstance(op.input.type, MemRefType):
            return

        # replace index -> x with index cast
        if isinstance(op.input.type, IndexType) and not isinstance(
            op.result.type, IndexType
        ):
            rewriter.replace_matched_op(index.CastsOp(op.input, op.result.type))

        if op.input.type == op.result.type and isinstance(op.input, BlockArgument):
            # replace x -> x with x
            rewriter.replace_matched_op((), (op.input,))


class ConvertAllocToIterArg(RewritePattern):
    def find_lowest_common_region(self, root: Region, uses: set[Use]) -> Region:
        """
        Find the lowest common region that includes all regions in the set. Ignores Alloc and Free .
        """

        assert len(uses) > 0, "uses should not be empty"

        paths = []
        for use in uses:
            # ignore Alloc and Free
            if isinstance(use.operation, exo.AllocOp) or isinstance(
                use.operation, exo.FreeOp
            ):
                continue

            path = []
            current = use.operation.parent_region()

            while current != root:
                path.append(current)
                current = current.parent_region()

            path.append(root)
            path.reverse()
            paths.append(path)

        # find the lowest common region
        for i in range(len(paths[0])):
            for j in range(1, len(paths)):
                if i >= len(paths[j]):
                    return paths[0][i - 1]

                if paths[j][i] != paths[0][i]:
                    return paths[0][i - 1]

        return paths[0][-1]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.ScalarOp, rewriter: PatternRewriter):
        # we cannot convert tensors
        if isinstance(op.result.type, MemRefType):
            return

        # we cannot convert values whose dealloc is not in the same region as the alloc
        for use in op.result.uses:
            if isinstance(use.operation, exo.FreeOp):
                if use.operation.parent_region() != op.parent_region():
                    return

        # find lowest common region
        lcr = self.find_lowest_common_region(op.parent_region(), op.result.uses)
        lcr_op = lcr.parent_op()

        # conditions:
        # - operation must exist
        # - operation must be a for op
        # - cannot be the same region as the alloc
        if (
            lcr_op is None
            or lcr is op.parent_region()
            or not isinstance(lcr_op, scf.ForOp)
        ):
            return

        # create new loop body with the alloc as an iter arg
        body = lcr_op.body.clone()
        body.blocks[0].insert_arg(op.result.type, len(body.blocks[0].args))

        # replace uses of the alloc with the iter arg
        for use in set(op.result.uses):
            if isinstance(use.operation, exo.FreeOp):
                continue

            use.operation.operands[use.index] = body.blocks[0].args[-1]

        # update loop op
        rewriter.replace_op(
            lcr_op,
            scf.ForOp(
                lcr_op.lb,
                lcr_op.ub,
                lcr_op.step,
                lcr_op.iter_args + (op.result,),
                body,
            ),
        )


class ConvertAllocToTensor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.AllocOp, rewriter: PatternRewriter):
        if isinstance(op.result.type, MemRefType):
            return

        rewriter.replace_matched_op(
            exo.AllocOp(
                op.mem.data, MemRefType(op.result.type, [1], NoneAttr(), op.mem)
            )
        )


class ConvertReadToTensor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.ReadOp, rewriter: PatternRewriter):
        if (
            not isinstance(op.input.type, MemRefType)
            or op.input.type.get_shape() != (1,)
            or len(op.indices) != 0
        ):
            return

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                exo.ReadOp(op.input, [zero_op.result], op.result.type),
            )
        )
        zero_op.result.name_hint = "c0"


class ConvertAssignToTensor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.AssignOp, rewriter: PatternRewriter):
        if (
            not isinstance(op.input.type, MemRefType)
            or op.input.type.get_shape() != (1,)
            or len(op.indices) != 0
        ):
            return

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                exo.AssignOp(op.input, [zero_op.result], op.value),
            )
        )
        zero_op.result.name_hint = "c0"


class ConvertReduceToTensor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.ReduceOp, rewriter: PatternRewriter):
        if (
            not isinstance(op.input.type, MemRefType)
            or op.input.type.get_shape() != (1,)
            or len(op.indices) != 0
        ):
            return

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                exo.ReduceOp(op.input, [zero_op.result], op.value),
            )
        )
        zero_op.result.name_hint = "c0"


class ConvertScalarFuncArgsToTensor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        for idx, arg in enumerate(op.args):
            # ignore tensor types
            if isinstance(arg.type, MemRefType):
                continue

            mutated = any(
                isinstance(use.operation, exo.AssignOp)
                and use.operation.input == arg
                or isinstance(use.operation, exo.ReduceOp)
                and use.operation.input == arg
                for use in arg.uses
            )

            # ignore unmutated scalar types, these can stay as is
            if not mutated:
                continue

            func_type = FunctionType.from_lists(
                (
                    *(arg.type for arg in op.args[:idx]),
                    MemRefType(arg.type, [1], NoneAttr()),
                    *(arg.type for arg in op.args[idx + 1 :]),
                ),
                op.function_type.outputs,
            )

            # rewrite function signature
            body = op.detach_region(op.body)
            new_arg = rewriter.insert_block_argument(
                body.block, idx, (MemRefType(arg.type, [1], NoneAttr()))
            )
            rewriter.replace_all_uses_with(
                arg,
                new_arg,
            )
            rewriter.erase_block_argument(arg, idx)

            rewriter.replace_matched_op(
                func.FuncOp(op.sym_name.data, func_type, body, op.sym_visibility)
            )


class ConvertScalarRefPass(ModulePass):
    name = "convert-scalar-ref"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertRedundantReads(),
                    ConvertAllocToTensor(),
                    ConvertReadToTensor(),
                    ConvertAssignToTensor(),
                    ConvertReduceToTensor(),
                    ConvertScalarFuncArgsToTensor(),
                ]
            ),
        ).rewrite_module(m)
