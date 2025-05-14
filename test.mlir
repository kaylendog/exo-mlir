builtin.module {
	func.func @load(%arr : memref<?x?xf32>, %0 : index, %1 : index) -> f32 {
		%res = memref.load %arr[%0, %1] : memref<?x?xf32>
		func.return %res : f32
	}
	func.func @store(%arr : memref<?x?xf32>, %0 : index, %1 : index, %val : f32) {
		memref.store %val, %arr[%0, %1] : memref<?x?xf32>
		func.return
	}
}
