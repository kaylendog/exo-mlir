import logging
import os
import contextlib
from collections.abc import Sequence
from pathlib import Path

from exo.API import Procedure
from exo.backend.LoopIR_compiler import find_all_subprocs
from exo.backend.mem_analysis import MemoryAnalysis
from exo.backend.parallel_analysis import ParallelAnalysis
from exo.backend.prec_analysis import PrecisionAnalysis
from exo.backend.win_analysis import WindowAnalysis
from exo.core.LoopIR import LoopIR
from exo.main import get_procs_from_module, load_user_code
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, Builtin
from xdsl.dialects import arith, func, memref, scf
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import (
    CommonSubexpressionElimination,
)
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass

from exomlir.dialects.exo import Exo
from exomlir.dialects.index import Index
from exomlir.generator import IRGenerator
from exomlir.platforms.avx2 import InlineAVX2Pass
from exomlir.rewrites.convert_scalar_ref import ConvertScalarRefPass
from exomlir.rewrites.convert_tensor_ref import ConvertTensorRefPass
from exomlir.rewrites.inline_memory_space import InlineMemorySpacePass
from exomlir.rewrites.tidy import TidyPass

logger = logging.getLogger("exo-mlir")


def context() -> Context:
    ctx = Context()
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(memref.MemRef)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(Exo)
    ctx.load_dialect(Index)
    return ctx


def analyze(p):
    """
    Perform the default Exo analysis on a procedure.
    """

    assert isinstance(p, LoopIR.proc)

    p = ParallelAnalysis().run(p)
    p = PrecisionAnalysis().run(p)
    p = WindowAnalysis().apply_proc(p)
    return MemoryAnalysis().run(p)


def compile_one(proc: Procedure) -> ModuleOp:
    """
    Compile a single procedure. This is an alias for `compile_many([proc])`.
    """
    if proc.is_instr():
        raise TypeError("Cannot compile an instr procedure.")
    return compile_many([proc])


def compile_many(
    library: Sequence[Procedure], print_exo=False, cache_exo=False
) -> ModuleOp:
    """
    Compile a list of procedures into a single MLIR module..
    """
    input_procedures = list(
        sorted(
            find_all_subprocs(
                [proc._loopir_proc for proc in library if not proc.is_instr()]
            ),
            key=lambda x: x.name,
        )
    )

    # ensure no duplicate procedures
    seen_procs = set()
    for proc in input_procedures:
        if proc.name in seen_procs:
            raise TypeError(f"multiple procs named {proc.name}")
        seen_procs.add(proc.name)

    # analyze procedures
    analyzed_procedures = [analyze(proc) for proc in input_procedures]

    # if print_exo is set, print the analyzed procedures
    if print_exo:
        for proc in analyzed_procedures:
            print(proc)

    if cache_exo:
        with open("exo_cache.txt", "w") as f:
            for proc in analyzed_procedures:
                f.write(str(proc) + "\n")
                f.write("\n")

    # generate MLIR
    return transform(context(), IRGenerator().generate(analyzed_procedures))


def compile_path(
    src: Path,
    dest: Path | None = None,
    print_exo: bool = False,
    cache_exo: bool = False,
):
    """
    Compile all procedures in a Python source file to a single MLIR module, and write it to a file.
    """
    if not src.exists():
        logger.error(f"{src} does not exist.")
        return

    if not src.is_file() or not src.suffix == ".py":
        logger.error(f"{src} is not a Python source file.")
        return

    logger.info(f"Compile[{src}] Destination: {dest}")

    # load user code and get procedures from exo
    # procedures tend to do a lot of printing, so we suppress stdout temporarily
    with contextlib.redirect_stdout(None):
        library = get_procs_from_module(load_user_code(src))  # type: list[Procedure]

    logger.info(f"Compile[{src}] Loaded {len(library)} procedure(s) from source")

    # invoke exo analysis
    assert isinstance(library, list)
    assert all(isinstance(proc, Procedure) for proc in library)

    module = compile_many(library, print_exo=print_exo, cache_exo=cache_exo)

    # print to stdout if no dest
    if not dest:
        print(module)
        return

    # write MLIR to file
    os.makedirs(dest.parent, exist_ok=True)
    dest.write_text(str(module))


def transform(ctx: Context, module: ModuleOp, lower_to_llvm=True) -> ModuleOp:
    """
    Apply transformations to an MLIR module.
    """

    InlineMemorySpacePass().apply(ctx, module)
    ConvertScalarRefPass().apply(ctx, module)
    ConvertTensorRefPass().apply(ctx, module)

    InlineAVX2Pass().apply(ctx, module)
    TidyPass().apply(ctx, module)

    CanonicalizePass().apply(ctx, module)
    CommonSubexpressionElimination().apply(ctx, module)

    # bail out if we are not lowering to LLVM
    if not lower_to_llvm:
        return module

    CanonicalizePass().apply(ctx, module)
    CommonSubexpressionElimination().apply(ctx, module)

    # ConvertMemRefToPtr(lower_func=True).apply(ctx, module)
    # ConvertPtrTypeOffsetsPass().apply(ctx, module)
    # ConvertPtrToLLVMPass().apply(ctx, module)
    # ConvertVectorPtrToLLVMPass().apply(ctx, module)

    ReconcileUnrealizedCastsPass().apply(ctx, module)

    return module
