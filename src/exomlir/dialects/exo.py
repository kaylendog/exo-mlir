from collections.abc import Sequence
from typing import Annotated, ClassVar, TypeAlias

from xdsl.dialects import arith, memref
from xdsl.dialects.builtin import (
    I32,
    DenseArrayBase,
    FlatSymbolRefAttrConstr,
    Float16Type,
    Float32Type,
    Float64Type,
    IntegerType,
    MemRefType,
    Signedness,
    StringAttr,
    SymbolRefAttr,
    TupleType,
    i64,
)
from xdsl.dialects.utils import split_dynamic_index_list
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.irdl import (
    AnyAttr,
    AnyOf,
    Attribute,
    AttrSizedOperandSegments,
    IRDLOperation,
    ParsePropInAttrDict,
    VarConstraint,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    var_operand_def,
)
from xdsl.printer import Printer

SizeType: TypeAlias = arith.IndexType
StrideType: TypeAlias = arith.IndexType
IndexType: TypeAlias = arith.IndexType

u8 = IntegerType(8, Signedness.UNSIGNED)
u16 = IntegerType(16, Signedness.UNSIGNED)

U8 = Annotated[IntegerType, u8]
U16 = Annotated[IntegerType, u16]

NumType: TypeAlias = AnyOf[Float16Type | Float32Type | Float64Type | U8 | U16 | I32]
IntType: TypeAlias = AnyOf[U8 | U16 | I32]

TensorType: TypeAlias = MemRefType[NumType]

IntervalType: TypeAlias = TupleType


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "exo.alloc"

    result = result_def()
    mem = prop_def(StringAttr)

    assembly_format = "$mem attr-dict `:` type($result)"

    def __init__(
        self,
        mem: str,
        result_type: MemRefType,
    ) -> None:
        super().__init__(
            operands=[],
            result_types=[result_type],
            properties={"mem": StringAttr(mem)},
        )


@irdl_op_definition
class FreeOp(IRDLOperation):
    name = "exo.free"

    input = operand_def()

    mem = prop_def(StringAttr)

    assembly_format = "$input $mem attr-dict `:` type($input)"

    def __init__(
        self,
        input: SSAValue | Operation,
        mem: str,
    ) -> None:
        super().__init__(
            operands=[input],
            result_types=[],
            properties={"mem": StringAttr(mem)},
        )


@irdl_op_definition
class AssignOp(IRDLOperation):
    name = "exo.assign"

    value = operand_def()
    input = operand_def()
    indices = var_operand_def(i64)
    sizes = var_operand_def(i64)
    static_sizes = prop_def(DenseArrayBase)

    assembly_format = "$value `,` $input `[` $indices `]` `,` `sizes` `:` `[` $sizes `]` `,` attr-dict `:` type($value) `,` type($input)"

    irdl_options = [AttrSizedOperandSegments(as_property=True), ParsePropInAttrDict()]

    def __init__(
        self,
        value: SSAValue | Operation,
        input: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        sizes: Sequence[SSAValue | int],
    ) -> None:
        static_sizes, dyn_sizes = split_dynamic_index_list(
            sizes, memref.SubviewOp.DYNAMIC_INDEX
        )
        super().__init__(
            operands=[value, SSAValue.get(input), indices, dyn_sizes],
            result_types=[],
            properties={
                "static_sizes": DenseArrayBase.create_dense_int(i64, static_sizes)
            },
        )


@irdl_op_definition
class ReduceOp(IRDLOperation):
    name = "exo.reduce"

    value = operand_def()
    input = operand_def()
    indices = var_operand_def(i64)
    sizes = var_operand_def(i64)
    static_sizes = prop_def(DenseArrayBase)

    assembly_format = "$value `,` $input `[` $indices `]` `,` `sizes` `:` `[` $sizes `]` `,` attr-dict `:` type($value) `,` type($input)"

    irdl_options = [AttrSizedOperandSegments(as_property=True), ParsePropInAttrDict()]

    def __init__(
        self,
        value: SSAValue | Operation,
        input: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        sizes: Sequence[SSAValue | int],
    ) -> None:
        static_sizes, dyn_sizes = split_dynamic_index_list(
            sizes, memref.SubviewOp.DYNAMIC_INDEX
        )

        super().__init__(
            operands=[value, SSAValue.get(input), indices, dyn_sizes],
            result_types=[],
            properties={
                "static_sizes": DenseArrayBase.create_dense_int(i64, static_sizes)
            },
        )


@irdl_op_definition
class ReadOp(IRDLOperation):
    name = "exo.read"

    input = operand_def()
    indices = var_operand_def(i64)
    sizes = var_operand_def(i64)
    static_sizes = prop_def(DenseArrayBase)
    result = result_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True), ParsePropInAttrDict()]

    def __init__(
        self,
        input: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        sizes: Sequence[SSAValue | int],
        result_type: Attribute,
    ) -> None:
        static_sizes, dyn_sizes = split_dynamic_index_list(
            sizes, memref.SubviewOp.DYNAMIC_INDEX
        )
        super().__init__(
            operands=[
                SSAValue.get(input),
                indices,
                dyn_sizes,
            ],
            result_types=[result_type],
            properties={
                "static_sizes": DenseArrayBase.create_dense_int(i64, static_sizes)
            },
        )

    def verify_(self):
        if isinstance(self.input.type, MemRefType) and isinstance(
            self.result.type, MemRefType
        ):
            if self.input.type != self.result.type:
                raise ValueError(
                    f"Input type {self.input.type} does not match result type {self.result.type}"
                )

        if isinstance(self.input.type, MemRefType) and not isinstance(
            self.result.type, MemRefType
        ):
            if self.input.type.element_type != self.result.type:
                raise ValueError(
                    f"Input element type {self.input.type.element_type} does not match result type {self.result.type}"
                )
        else:
            if len(self.indices) > 0:
                raise ValueError(
                    f"Expected no indices for non-memref input, but got {self.indices}"
                )

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print(self.input)
        if len(self.indices) > 0:
            printer.print("[")
            for i, index in enumerate(self.indices):
                if i > 0:
                    printer.print(", ")
                printer.print(index)
            printer.print("]")
        printer.print(" -> ")
        printer.print(self.result.type)


@irdl_op_definition
class IntervalOp(IRDLOperation):
    name = "exo.interval"

    start = operand_def(IndexType)
    end = operand_def(IndexType)
    result = result_def(IntervalType)

    assembly_format = "$start `,` $end attr-dict `:` type($result)"

    def __init__(
        self,
        start: SSAValue | Operation,
        end: SSAValue | Operation,
    ) -> None:
        super().__init__(
            operands=[SSAValue.get(start), SSAValue.get(end)],
            result_types=[TupleType([start.type, end.type])],
        )


@irdl_op_definition
class WindowOp(IRDLOperation):
    T: ClassVar = VarConstraint("T", AnyAttr())

    name = "exo.window"

    input = operand_def(
        MemRefType.constr(element_type=AnyAttr()),
    )
    indices = var_operand_def()
    input_sizes = var_operand_def()
    static_input_sizes = prop_def(DenseArrayBase)
    output_sizes = var_operand_def()
    static_output_sizes = prop_def(DenseArrayBase)

    result = result_def(
        MemRefType.constr(element_type=T),
    )

    irdl_options = [AttrSizedOperandSegments(as_property=True), ParsePropInAttrDict()]

    def __init__(
        self,
        input: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        input_sizes: Sequence[SSAValue | Operation | int],
        output_sizes: Sequence[SSAValue[Attribute] | int],
        result_type: MemRefType,
    ) -> None:
        static_input_sizes, dyn_input_sizes = split_dynamic_index_list(
            input_sizes, memref.SubviewOp.DYNAMIC_INDEX
        )
        static_output_sizes, dyn_output_sizes = split_dynamic_index_list(
            output_sizes, memref.SubviewOp.DYNAMIC_INDEX
        )

        super().__init__(
            operands=[SSAValue.get(input), indices, dyn_input_sizes, dyn_output_sizes],
            result_types=[result_type],
            properties={
                "static_input_sizes": DenseArrayBase.create_dense_int(
                    i64, static_input_sizes
                ),
                "static_output_sizes": DenseArrayBase.create_dense_int(
                    i64, static_output_sizes
                ),
            },
        )


@irdl_op_definition
class InstrOp(IRDLOperation):
    name = "exo.instr"

    arguments = var_operand_def()
    callee = prop_def(FlatSymbolRefAttrConstr)

    assembly_format = "$callee attr-dict `(` $arguments `)` `:` type($arguments)"

    def __init__(
        self,
        callee: str | SymbolRefAttr,
        arguments: Sequence[SSAValue | Operation],
    ) -> None:
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)
        super().__init__(
            operands=[arguments], result_types=[], properties={"callee": callee}
        )


@irdl_op_definition
class ExternOp(IRDLOperation):
    name = "exo.extern"

    arguments = var_operand_def()
    result = result_def()

    callee = prop_def(FlatSymbolRefAttrConstr)

    assembly_format = (
        "$callee attr-dict `(` $arguments `)` `:` type($arguments) `->` type($result)"
    )

    def __init__(
        self,
        callee: str | SymbolRefAttr,
        arguments: Sequence[SSAValue | Operation],
        result_type: Attribute,
    ) -> None:
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)
        super().__init__(
            operands=[arguments],
            result_types=[result_type],
            properties={"callee": callee},
        )


Exo = Dialect(
    "exo",
    [AssignOp, ReduceOp, ReadOp, WindowOp, IntervalOp, InstrOp],
    [],
)
