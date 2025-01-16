import logging
import os
import sys
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
from xdsl.parser import ModuleOp

from exomlir.generator import IRGenerator

logger = logging.getLogger("exo-mlir")


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


def compile_many(library: Sequence[Procedure]) -> ModuleOp:
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

    # generate MLIR
    return IRGenerator().generate(analyzed_procedures)


def compile_path(src: Path, dest: Path | None = None):
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

    # load user code and get procedures from exo - procedures tend to erase stdout, so we save it
    stdout = sys.stdout
    library = get_procs_from_module(load_user_code(src))  # type: list[Procedure]

    logger.info(f"Compile[{src}] Loaded {len(library)} procedure(s) from source")

    # invoke exo analysis
    assert isinstance(library, list)
    assert all(isinstance(proc, Procedure) for proc in library)

    module = compile_many(library)
    sys.stdout = stdout

    # print to stdout if no dest
    if not dest:
        print(module)
        return

    # write MLIR to file
    os.makedirs(dest.parent, exist_ok=True)
    dest.write_text(str(module))
