from typing import ClassVar

from xdsl.dialects import arith
from xdsl.dialects.builtin import (
    I1,
    IntegerAttr,
    VectorType,
    i32,
)
from xdsl.dialects.llvm import LLVMCompatibleFloatConstraint, LLVMPointerType
from xdsl.ir import Dialect
from xdsl.irdl import (
    Attribute,
    IRDLOperation,
    Operation,
    ParsePropInAttrDict,
    SSAValue,
    VarConstraint,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
)


@irdl_op_definition
class FAbsOp(IRDLOperation):
    T: ClassVar = VarConstraint("T", LLVMCompatibleFloatConstraint)

    name = "llvm.intr.fabs"

    input = operand_def(T)
    result = result_def(T)

    assembly_format = (
        "`(` operands `)` attr-dict `:` functional-type(operands, results)"
    )

    irdl_options = [ParsePropInAttrDict()]

    def __init__(self, input: Operation | SSAValue, result_type: Attribute):
        super().__init__(operands=[input], result_types=[result_type])


@irdl_op_definition
class MaskedStoreOp(IRDLOperation):
    name = "llvm.intr.masked.store"

    value = operand_def(LLVMCompatibleFloatConstraint)
    data = operand_def(LLVMPointerType)
    mask = operand_def(I1 | VectorType[I1])
    alignment = prop_def(IntegerAttr[i32])

    assembly_format = "$value `,` $data `,` $mask attr-dict `:` type($value) `,` type($mask) `into` type($data)"

    irdl_options = [ParsePropInAttrDict()]

    def __init__(
        self,
        value: Operation | SSAValue,
        data: Operation | SSAValue,
        mask: Operation | SSAValue,
        alignment: int = 32,
    ):
        super().__init__(
            operands=[value, data, mask],
            result_types=[],
            properties={
                "alignment": IntegerAttr.from_int_and_width(alignment, 32),
            },
        )


LLVMIntrinsics = Dialect(
    "llvm.intr",
    [FAbsOp, MaskedStoreOp],
    [],
)
