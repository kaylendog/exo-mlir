builtin.module {
  func.func @add_one(%0 : memref<16xf32, strided<[1]>, "DRAM">) {
    %1 = arith.constant 0 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = arith.constant 16 : i32
    %4 = arith.index_cast %3 : i32 to index
    %5 = arith.constant 1 : index
    %6 = arith.constant 1.000000e+00 : f32
    scf.for %7 = %2 to %4 step %5 {
      %bytes_per_element = ptr_xdsl.type_offset f32 : index
      %scaled_pointer_offset = arith.muli %7, %bytes_per_element : index
      %8 = ptr_xdsl.to_ptr %0 : memref<16xf32, strided<[1]>, "DRAM"> -> !ptr_xdsl.ptr
      %offset_pointer = ptr_xdsl.ptradd %8, %scaled_pointer_offset : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
      %9 = ptr_xdsl.load %offset_pointer : !ptr_xdsl.ptr -> f32
      %10 = arith.subi %9, %6 : index
      %bytes_per_element_1 = ptr_xdsl.type_offset f32 : index
      %scaled_pointer_offset_1 = arith.muli %7, %bytes_per_element_1 : index
      %11 = ptr_xdsl.to_ptr %0 : memref<16xf32, strided<[1]>, "DRAM"> -> !ptr_xdsl.ptr
      %offset_pointer_1 = ptr_xdsl.ptradd %11, %scaled_pointer_offset_1 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
      ptr_xdsl.store %10, %offset_pointer_1 : index, !ptr_xdsl.ptr
    }
    func.return
  }
}
