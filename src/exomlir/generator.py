from __future__ import annotations

from typing import TypeAlias

from exo.API import Sym
from exo.core.LoopIR import LoopIR, T
from xdsl.builder import Builder
from xdsl.dialects.arith import (
    AddfOp,
    AddiOp,
    AndIOp,
    CmpfOp,
    CmpiOp,
    ConstantOp,
    DivfOp,
    DivSIOp,
    IndexCastOp,
    MulfOp,
    MuliOp,
    NegfOp,
    OrIOp,
    RemSIOp,
    SubfOp,
    SubiOp,
)
from xdsl.dialects.builtin import (
    BoolAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    FloatAttr,
    FunctionType,
    IndexType,
    IntAttr,
    IntegerAttr,
    MemRefType,
    ModuleOp,
    f16,
    f32,
    f64,
    i1,
    i8,
    i16,
    i32,
)
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.dialects.memref import AllocOp, DeallocOp, LoadOp, StoreOp
from xdsl.dialects.scf import ForOp, IfOp, YieldOp
from xdsl.dialects.test import TestOp
from xdsl.ir import Block, BlockArgument, OpResult, Region, SSAValue
from xdsl.utils.scoped_dict import ScopedDict

MemRefTypeF16: TypeAlias = MemRefType[Float16Type]
MemRefTypeF32: TypeAlias = MemRefType[Float32Type]
MemRefTypeF64: TypeAlias = MemRefType[Float64Type]


INTEGER_CMP_TABLE = {
    "==": "eq",
    "!=": "ne",
    "<": "slt",
    "<=": "sle",
    ">": "sgt",
    ">=": "sge",
}

FLOAT_CMP_TABLE = {
    "==": "oeq",
    "!=": "one",
    "<": "olt",
    "<=": "ole",
    ">": "ogt",
    ">=": "oge",
}


class IRGeneratorError(Exception):
    pass


class IRGenerator:
    module: ModuleOp
    builder: Builder

    symbol_table: ScopedDict[str, SSAValue] | None = None

    seen_procs: set[str] = set()

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder.at_end(self.module.body.blocks[0])

    def with_empty_scope(self):
        """
        Return this IRGenerator with an empty symbol table.
        """
        self.symbol_table = ScopedDict()
        return self

    def declare_arg(self, sym: Sym, arg: BlockArgument) -> BlockArgument:
        """
        Declare a symbol in the symbol table.
        """
        assert self.symbol_table is not None
        self.declare_value(sym, arg)
        return arg

    def declare_value(self, sym: Sym, value: SSAValue) -> SSAValue:
        """
        Declare a value in the symbol table.
        """
        assert self.symbol_table is not None
        self.symbol_table[sym.name()] = value
        return value

    def _with_test_op(self, sym: Sym, type):
        assert self.symbol_table is not None
        op = TestOp(result_types=[self.get_type(type)])
        self.builder.insert(op)
        self.symbol_table[sym.name()] = op.res[0]
        return self

    def generate(self, procs) -> ModuleOp:
        for proc in procs:
            self.generate_procedure(proc)

        # verify module
        # TODO: none of the operations actually implement verify_()
        try:
            self.module.verify()
        except Exception as e:
            print("module verification failed: ", e)
            raise

        return self.module

    def get_sym(self, sym: Sym) -> SSAValue:
        """Get the SSAValue for a symbol."""
        assert self.symbol_table is not None

        if sym.name() not in self.symbol_table:
            raise IRGeneratorError(f"Unknown symbol {sym.name()}")

        return self.symbol_table[sym.name()]

    def cast_to_index(self, value: SSAValue) -> SSAValue:
        # must not cast if already an index
        if isinstance(value.type, IndexType):
            return value
        cast = IndexCastOp(value, IndexType())
        self.builder.insert(cast)
        return cast.result

    def generate_procedure(self, procedure):
        if procedure.name in self.seen_procs:
            return

        self.seen_procs.add(procedure.name)

        parent_builder = self.builder
        self.symbol_table = ScopedDict[str, SSAValue]()

        # initialise function block
        block = Block(arg_types=[self.get_type(arg.type) for arg in procedure.args])
        self.builder = Builder.at_end(block)

        # add arguments to symbol table
        for idx, (proc_arg, block_arg) in enumerate(zip(procedure.args, block.args)):
            self.declare_arg(proc_arg.name, block_arg)

        # generate function body
        self.generate_stmt_list(procedure.body)
        self.builder.insert(ReturnOp())

        # cleanup
        self.symbol_table = None
        self.builder = parent_builder

        input_types = [self.get_type(arg.type) for arg in procedure.args]
        func_type = FunctionType.from_lists(input_types, [])

        # insert procedure into module
        self.builder.insert(FuncOp(procedure.name, func_type, Region(block)))

    def generate_stmt_list(self, stmts):
        """Generate a list of statements."""
        for stmt in stmts:
            self.generate_stmt(stmt)

    def generate_stmt(self, stmt):
        if isinstance(stmt, LoopIR.Assign):
            self.generate_assign_stmt(stmt)
        elif isinstance(stmt, LoopIR.Reduce):
            self.generate_reduce_stmt(stmt)
        elif isinstance(stmt, LoopIR.WriteConfig):
            self.generate_write_config_stmt(stmt)
        elif isinstance(stmt, LoopIR.Pass):
            # do nothing!!
            pass
        elif isinstance(stmt, LoopIR.If):
            self.generate_if_stmt(stmt)
        elif isinstance(stmt, LoopIR.For):
            self.generate_for_stmt(stmt)
        elif isinstance(stmt, LoopIR.Alloc):
            self.generate_alloc_stmt(stmt)
        elif isinstance(stmt, LoopIR.Free):
            self.generate_free_stmt(stmt)
        elif isinstance(stmt, LoopIR.Call):
            # TODO: call stmts are not supported yet
            pass
        elif isinstance(stmt, LoopIR.Window):
            self.generate_window_stmt(stmt)
        else:
            raise IRGeneratorError(f"Unknown statement {stmt}")

    def generate_assign_stmt(self, assign):
        idx = self.generate_expr_list(assign.idx)
        idx = [self.cast_to_index(i) for i in idx]
        rhs = self.generate_expr(assign.rhs)
        self.builder.insert(StoreOp(operands=[rhs, self.get_sym(assign.name), idx]))

    def generate_reduce_stmt(self, reduce):
        idx = self.generate_expr_list(reduce.idx)
        idx = [self.cast_to_index(i) for i in idx]
        rhs = self.generate_expr(reduce.rhs)

        memref = self.get_sym(reduce.name)

        # load value from memory, add rhs, store back - could use AtomicRMWOp here?
        load = LoadOp(operands=[memref, idx], result_types=[rhs.type])
        inc = AddfOp(load.res, rhs, result_type=rhs.type)
        store = StoreOp(operands=[inc.result, memref, idx])

        self.builder.insert(load)
        self.builder.insert(inc)
        self.builder.insert(store)

    def generate_write_config_stmt(self, write_config):
        # rhs = self.generate_expr(write_config.rhs)
        # self.builder.insert(WriteConfigOp(write_config.name, write_config.field, rhs))
        raise NotImplementedError

    def generate_if_stmt(self, if_stmt):
        cond = self.generate_expr(if_stmt.cond)

        parent_builder = self.builder

        # construct true_block
        true_block = Block()
        self.builder = Builder.at_end(true_block)
        self.generate_stmt_list(if_stmt.body)
        self.builder.insert(YieldOp())

        # construct false_block
        false_block = Block()
        self.builder = Builder.at_end(false_block)
        self.generate_stmt_list(if_stmt.orelse)
        self.builder.insert(YieldOp())

        # cleanup and construct
        self.builder = parent_builder
        self.builder.insert(IfOp(cond, [], Region(true_block), Region(false_block)))

    def generate_for_stmt(self, for_stmt):
        lo = self.generate_expr(for_stmt.lo)
        lo = self.cast_to_index(lo)
        hi = self.generate_expr(for_stmt.hi)
        hi = self.cast_to_index(hi)
        step = ConstantOp(IntegerAttr(1, IndexType()))
        self.builder.insert(step)

        parent_builder = self.builder
        parent_scope = self.symbol_table

        # construct loop block
        loop_block = Block(
            # TODO: this should be inferred from lo and hi
            arg_types=[IndexType()],
        )
        self.builder = Builder.at_end(loop_block)
        self.symbol_table = ScopedDict(parent_scope)

        # add loop variable to symbol table
        self.declare_arg(for_stmt.iter, loop_block.args[0])

        # generate loop body
        self.generate_stmt_list(for_stmt.body)
        self.builder.insert(YieldOp())

        # cleanup and construct
        self.symbol_table = parent_scope
        self.builder = parent_builder

        self.builder.insert(ForOp(lo, hi, step.result, [], Region(loop_block)))

    def generate_alloc_stmt(self, alloc):
        op = AllocOp([], [], result_type=self.get_type(alloc.type))
        self.builder.insert(op)
        self.declare_value(alloc.name, op.results[0])
        return op.results[0]

    def generate_free_stmt(self, free):
        self.builder.insert(
            DeallocOp(operands=[self.get_sym(free.name)], result_types=[])
        )

    def generate_call_stmt(self, call):
        # TODO: procedure generation should be top-level, then call should simply use a SymRefAttr to refer to the procedure
        self.generate_procedure(call.f)
        args = [self.generate_expr(arg) for arg in call.args]
        self.builder.insert(CallOp(call.f.name, args, []))

    # def generate_window_stmt(self, window):
    #     rhs = self.generate_expr(window.rhs)
    #     self.builder.insert(WindowStmtOp(self.symbol(window.name), rhs))

    def generate_expr_list(self, exprs) -> list[OpResult | SSAValue]:
        return [self.generate_expr(expr) for expr in exprs]

    def generate_expr(self, expr) -> OpResult | SSAValue:
        if isinstance(expr, LoopIR.Read):
            return self.generate_read_expr(expr)
        elif isinstance(expr, LoopIR.Const):
            return self.generate_const_expr(expr)
        elif isinstance(expr, LoopIR.BinOp):
            return self.generate_binop_expr(expr)
        else:
            raise IRGeneratorError(f"Unknown expression {expr}")

    def generate_read_expr(self, read):
        idx = self.generate_expr_list(read.idx)
        idx = [self.cast_to_index(i) for i in idx]

        operand = self.get_sym(read.name)

        # if operand is a tensor, we need to load from memory
        if isinstance(operand.type, MemRefType):
            read = LoadOp(
                operands=[self.get_sym(read.name), idx],
                result_types=[self.get_type(read.type)],
            )
            self.builder.insert(read)

            return read.res

        # otherwise, we can just return the operand
        else:
            return operand

    def generate_const_expr(self, const):
        type = self.get_type(const.type)

        # construct attribute depending on type
        if type in [f16, f32, f64]:
            attr = FloatAttr(const.val, type)
        elif type in [i8, i16, i32]:
            attr = IntegerAttr(const.val, type)
        elif type == i1:
            attr = BoolAttr(const.val, i1)
        else:
            raise IRGeneratorError(f"Unknown type {type} passed to Const")

        const = ConstantOp(attr, self.get_type(const.type))
        self.builder.insert(const)
        return const.result

    def generate_usub_expr(self, usub):
        expr = self.generate_expr(usub.arg)
        # float case
        if self.get_type(usub.type) in [f16, f32, f64]:
            usub = NegfOp(expr)
        # integer case
        elif self.get_type(usub.type) in [i8, i16, i32]:
            zero = ConstantOp(IntegerAttr(0, self.get_type(usub.type)))
            usub = SubiOp(zero.result, expr, result_type=self.get_type(usub.type))
            self.builder.insert(zero)
        else:
            raise IRGeneratorError(f"Bad type {type} passed to USub")

        self.builder.insert(usub)
        return usub.result

    def generate_binop_expr(self, binop):
        type = self.get_type(binop.type)

        if type in [f16, f32, f64]:
            return self.generate_binop_expr_float(binop)
        elif type in [i8, i16, i32]:
            return self.generate_binop_expr_int(binop)
        elif type == i1:
            return self.generate_binop_expr_cmp(binop)
        else:
            raise IRGeneratorError(f"Unknown type '{type.name}'")

    def generate_binop_expr_float(self, binop):
        lhs = self.generate_expr(binop.lhs)
        rhs = self.generate_expr(binop.rhs)
        type = self.get_type(binop.type)

        if binop.op == "+":
            binop = AddfOp(lhs, rhs, result_type=type)
        elif binop.op == "-":
            binop = SubfOp(lhs, rhs, result_type=type)
        elif binop.op == "*":
            binop = MulfOp(lhs, rhs, result_type=type)
        elif binop.op == "/":
            binop = DivfOp(lhs, rhs, result_type=type)
        else:
            raise IRGeneratorError(f"Unknown binop {binop.op}")

        self.builder.insert(binop)
        return binop.result

    def generate_binop_expr_int(self, binop):
        lhs = self.generate_expr(binop.lhs)
        rhs = self.generate_expr(binop.rhs)
        type = self.get_type(binop.type)

        if binop.op == "+":
            binop = AddiOp(lhs, rhs, result_type=type)
        elif binop.op == "-":
            binop = SubiOp(lhs, rhs, result_type=type)
        elif binop.op == "*":
            binop = MuliOp(lhs, rhs, result_type=type)
        elif binop.op == "/":
            binop = DivSIOp(lhs, rhs, result_type=type)
        elif binop.op == "%":
            binop = RemSIOp(lhs, rhs, result_type=type)
        else:
            raise IRGeneratorError(f"Unknown binop {binop.op}")

        self.builder.insert(binop)
        return binop.result

    def generate_binop_expr_cmp(self, binop):
        lhs = self.generate_expr(binop.lhs)
        rhs = self.generate_expr(binop.rhs)

        # boolean operations
        if lhs.type == i1:
            if binop.op == "and":
                binop = AndIOp(lhs, rhs)
            elif binop.op == "or":
                binop = OrIOp(lhs, rhs)
            else:
                raise IRGeneratorError(f"Unknown boolean operator '{binop.op}'")
        # cmpi
        elif lhs.type in [i8, i16, i32]:
            op = INTEGER_CMP_TABLE[binop.op]
            if op is None:
                raise IRGeneratorError(
                    f"Unknown integer comparison operator '{binop.op}'"
                )

            binop = CmpiOp(lhs, rhs, op)
        # cmpf
        else:
            op = FLOAT_CMP_TABLE[binop.op]
            if op is None:
                raise IRGeneratorError(
                    f"Unknown float comparison operator '{binop.op}'"
                )

            binop = CmpfOp(lhs, rhs, op)

        self.builder.insert(binop)
        return binop.result

    def generate_extern_expr(self, extern):
        args = self.generate_expr_list(extern.args)
        extern = CallOp(extern.f.name, args, [])
        self.builder.insert(extern)
        return extern.res

    def generate_window_expr(self, window):
        raise NotImplementedError()

    def generate_stride_expr(self, stride):
        raise NotImplementedError()

    def generate_read_config_expr(self, read_config):
        raise NotImplementedError()

    def get_type(self, t):
        # mlir
        if isinstance(t, SSAValue):
            return t.type
        # exo
        if isinstance(t, T.F16):
            return f16
        elif isinstance(t, T.F32) or isinstance(t, T.Num):
            return f32
        elif isinstance(t, T.F64):
            return f64
        elif isinstance(t, T.INT8) or isinstance(t, T.UINT8):
            return i8
        elif isinstance(t, T.UINT16):
            return i16
        elif (
            isinstance(t, T.INT32)
            or isinstance(t, T.Int)
            or isinstance(t, T.Index)
            or isinstance(t, T.Size)
        ):
            return i32
        elif isinstance(t, T.Bool):
            return i1
        elif isinstance(t, T.Tensor):
            inner = self.get_type(t.type)
            if inner == f16:
                return MemRefTypeF16(f16, self.get_shape(t))
            elif inner == f32:
                return MemRefTypeF32(f32, self.get_shape(t))
            elif inner == f64:
                return MemRefTypeF64(f64, self.get_shape(t))
            else:
                raise IRGeneratorError(f"Unknown tensor type '{t}'")
        else:
            raise IRGeneratorError(f"Unknown type '{t}'")

    def get_shape(self, type) -> list[IntegerAttr]:
        assert isinstance(type, T.Tensor)

        def attr_from_expr(expr):
            if isinstance(expr, LoopIR.Const):
                return IntAttr(expr.val)
            elif isinstance(expr, LoopIR.Read):
                return IntAttr(-1)
            else:
                raise IRGeneratorError(f"Invalid shape argument {expr}")

        return [attr_from_expr(expr) for expr in type.shape()]
