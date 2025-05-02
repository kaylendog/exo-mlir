from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, MemRefType
from xdsl.dialects import scf
from xdsl.passes import ModulePass
from xdsl.ir import Region, Use

from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
    RewritePattern,
    op_type_rewrite_pattern,
)


from exomlir.dialects import exo


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
    def match_and_rewrite(self, op: exo.AllocOp, rewriter: PatternRewriter):
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

        print(f"Converting {op} to iter arg of {lcr_op}")

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


class LowerIterArg(ModulePass):
    name = "lower-iter-args"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertAllocToIterArg(),
                ]
            )
        ).rewrite_module(m)
