from collections.abc import Sequence
from typing import Annotated, ClassVar, TypeAlias
from xdsl.dialects.builtin import (
    IntegerType,
    Float16Type,
    Float32Type,
    Float64Type,
    I32,
    Signedness,
    StringAttr,
    MemRefType,
    TupleType,
)
from xdsl.ir import Dialect, SSAValue, Operation


from xdsl.irdl import (
    IRDLOperation,
    AnyAttr,
    VarConstraint,
    Attribute,
    AnyOf,
    irdl_op_definition,
    operand_def,
    result_def,
    prop_def,
    var_operand_def,
)
from xdsl.dialects import arith


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

    input = operand_def()
    indices = var_operand_def(IndexType)
    value = operand_def()

    def __init__(
        self,
        input: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        value: SSAValue | Operation,
    ) -> None:
        super().__init__(
            operands=[SSAValue.get(input), indices, value], result_types=[]
        )


@irdl_op_definition
class ReduceOp(IRDLOperation):
    name = "exo.reduce"

    input = operand_def()
    indices = var_operand_def(IndexType)
    value = operand_def()

    def __init__(
        self,
        input: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        value: SSAValue | Operation,
    ) -> None:
        super().__init__(
            operands=[SSAValue.get(input), indices, value], result_types=[]
        )


@irdl_op_definition
class ReadOp(IRDLOperation):
    name = "exo.read"

    input = operand_def()
    indices = var_operand_def()

    result = result_def()

    def __init__(
        self,
        input: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[SSAValue.get(input), indices], result_types=[result_type]
        )

    def verify_(self):
        if isinstance(self.input.type, MemRefType):
            if self.input.type.element_type != self.result.type:
                raise ValueError(
                    f"Input type {self.input.type} does not match result type {self.result.type}"
                )
        else:
            if len(self.indices) > 0:
                raise ValueError(
                    f"Expected no indices for non-memref input, but got {self.indices}"
                )


@irdl_op_definition
class IntervalOp(IRDLOperation):
    name = "exo.interval"

    start = operand_def(IndexType)
    end = operand_def(IndexType)

    result = result_def(IntervalType)

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

    result = result_def(
        MemRefType.constr(element_type=T),
    )

    def __init__(
        self,
        memref: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        result_type: MemRefType,
    ) -> None:
        super().__init__(
            operands=[SSAValue.get(memref), indices], result_types=[result_type]
        )


Exo = Dialect(
    "exo",
    [
        AssignOp,
        ReduceOp,
        ReadOp,
        WindowOp,
        IntervalOp,
    ],
    [],
)
