from tensor import Tensor

alias F32Tensor = Tensor[DType.float32]

fn reg_type_check[T: AnyRegType]():
    print("Good!")


fn build_tuple() -> Tuple[F32Tensor, F32Tensor]:
    return F32Tensor(), F32Tensor()


fn main():
    reg_type_check[F32Tensor]()
    var tensors = build_tuple()
    var tensor_1 = tensors.get[0, F32Tensor]()

