# /* Inference for Llama-2 Transformer model in pure C */
# Port from C to Mojo


@register_passable
struct Config:
    alias VAR_COUNT = 7
    alias type = Int32
    alias SIZE = Self.VAR_COUNT * sizeof[Self.type]()

    var dim: Int
    var hidden_dim: Int
    var n_layers: Int
    var n_heads: Int
    var n_kv_heads: Int
    var vocab_size: Int
    var seq_len: Int

    var head_size: Int
    var kv_dim: Int
    var kv_mul: Int

    fn __init__(inout self, inout file_handle: FileHandle) raises:
        """Assume that the file contains all information in binary format contiguous in memory.
        """
        var raw_data = file_handle.read_bytes(Config.SIZE)
        var config = Tensor(raw_data).astype[DType.int32]()

        # Vocab size will be negative if we are using shared weights
        self.dim = config[0].to_int()
        self.hidden_dim = config[1].to_int()
        self.n_layers = config[2].to_int()
        self.n_heads = config[3].to_int()
        self.n_kv_heads = config[4].to_int()
        self.vocab_size = (
            config[5].to_int() if config[5].to_int() > 0 else -config[5].to_int()
        )
        self.seq_len = config[6].to_int()

        # Calculated values
        self.head_size = self.dim // self.n_heads
        self.kv_dim = (self.dim * self.n_kv_heads) // self.n_heads
        self.kv_mul = self.n_heads // self.n_kv_heads


from tensor import TensorShape


fn read_weights(inout f: FileHandle, inout readed: Int, *dims: Int) raises -> TensorF32:
    var shape = TensorShape(dims)
    var raw = f.read_bytes(shape.num_elements() * sizeof[DType.float32]())
    readed += shape.num_elements() * sizeof[DType.float32]()
    var tensor = Tensor(raw).astype[DType.float32]()
    var final_tensor = tensor.reshape(shape)
    return final_tensor


struct TransformerWeights:
    alias type = DType.float32
    var token_embedding_table: Tensor[Self.type]
    var rms_att_weight: Tensor[Self.type]
    var rms_ffn_weight: Tensor[Self.type]
    var wq: Tensor[Self.type]
    var wk: Tensor[Self.type]
    var wv: Tensor[Self.type]
    var wo: Tensor[Self.type]
    var w1: Tensor[Self.type]
    var w2: Tensor[Self.type]
    var w3: Tensor[Self.type]
    var rms_final_weight: Tensor[Self.type]
    var wcls: Tensor[Self.type]

    fn __init__(inout self, inout file: FileHandle, config: Config) raises:
        var readed = Config.SIZE
        self.token_embedding_table = read_weights(
            file, readed, config.vocab_size, config.dim
        )
        self.rms_att_weight = read_weights(file, readed, config.n_layers, config.dim)
        self.wq = read_weights(file, readed, config.n_layers, config.dim, config.dim)
        self.wk = read_weights(file, readed, config.n_layers, config.kv_dim, config.dim)
        self.wv = read_weights(file, readed, config.n_layers, config.kv_dim, config.dim)
        self.wo = read_weights(file, readed, config.n_layers, config.dim, config.dim)
        self.rms_ffn_weight = read_weights(file, readed, config.n_layers, config.dim)
        self.w1 = read_weights(
            file, readed, config.n_layers, config.hidden_dim, config.dim
        )
        self.w2 = read_weights(
            file, readed, config.n_layers, config.dim, config.hidden_dim
        )
        self.w3 = read_weights(
            file, readed, config.n_layers, config.hidden_dim, config.dim
        )
        self.rms_final_weight = read_weights(file, readed, config.dim)
        _ = file.read_bytes(config.seq_len * config.head_size / 2)
        _ = file.read_bytes(config.seq_len * config.head_size / 2)
        self.wcls = read_weights(file, readed, config.dim, config.vocab_size)
        print("readed:", readed, ". Checkpoint size: ", readed // 1024 // 1024, "MB")


fn foo[T: AnyRegType]() -> None:
    ...


fn m():
    foo[TensorF32]()


from tensor import Tensor

alias TensorF32 = Tensor[DType.float32]
from collections import Optional


@value
struct RunState:
    var x: TensorF32
    var xb: TensorF32
    var xb2: TensorF32
    var hb: TensorF32
    var hb2: TensorF32
    var q: TensorF32
    var k: Optional[TensorF32]
    var v: Optional[TensorF32]
    var att: TensorF32
    var logits: TensorF32
    var key_cache: TensorF32
    var value_cache: TensorF32

    fn __init__(inout self, config: Config):
        var kv_dim = (config.dim * config.n_kv_heads) // config.n_heads
        self.x = TensorF32(config.dim)
        self.xb = TensorF32(config.dim)
        self.xb2 = TensorF32(config.dim)
        self.hb = TensorF32(config.hidden_dim)
        self.hb2 = TensorF32(config.hidden_dim)
        self.q = TensorF32(config.dim)
        self.key_cache = TensorF32(config.n_layers * config.seq_len * kv_dim)
        self.value_cache = TensorF32(config.n_layers * config.seq_len * kv_dim)
        # self.k = TensorF32(config.dim)
        # self.v = TensorF32(config.dim)
        self.att = TensorF32(config.n_heads * config.seq_len)
        self.logits = TensorF32(config.vocab_size)

        # not initialized
        self.k = None
        self.v = None


struct Transformer:
    var config: Config
    var weights: TransformerWeights
    var state: RunState
    var fd: Int
    var data: Pointer[Float32]
    var file_size: Int


struct Mode:
    alias type = StringLiteral
    var _value: Self.type
    alias Generate: Mode = "generate"
    alias Chat: Mode = "chat"

    fn __init__(inout self, value: StringLiteral):
        self._value = value


struct TokenIndex:
    var name: String
    var id: Int


struct Tokenizer:
    var vocab: String
    var vocab_scores: Float32
    var sorted_vocab: TokenIndex
    var vocab_size: Int
    var max_token_length: UInt32
    var byte_pieces: SIMD[DType.uint8, 512]

    fn __init__[
        path: PathLike
    ](inout self, tokenizer_path: path, vocab_size: Int) raises:
        self.vocab_size = vocab_size

        self.byte_pieces = SIMD[DType.uint8, 512]()
        for i in range(512):
            self.byte_pieces[i] = ord(str(i))
            self.byte_pieces[i + 1] = ord("\0")

        with open(tokenizer_path, "rb") as f:
            var tk_len = f.read_bytes(sizeof[UInt32]())
            # self.max_token_length = tk_len.data.v
            # self.vocab = f.read()
            # self.vocab_scores = f.read()
            # self.sorted_vocab = f.read()
            # self.max_token_length = f.read(


from sys import argv


fn error_usage():
    print("Usage:   run <checkpoint> [options]\n")
    print('Example: run model.bin -n 256 -i "Once upon a time"\n')
    print("Options:\n")
    print("  -t <float>  temperature in [0,inf], default 1.0\n")
    print("  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n")
    print("  -s <int>    random seed, default time(NULL)\n")
    print("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n")
    print("  -i <string> input prompt\n")
    print("  -z <string> optional path to custom tokenizer\n")
    print("  -m <string> mode: generate|chat, default: generate\n")
    print("  -y <string> (optional) system prompt in chat mode\n")


from builtin.simd import SIMD


fn cast_to_float(string: String) -> Optional[Float64]:
    from builtin.string import atol

    try:
        var pos = string.find(".")
        var int_val = atol(string.replace(".", ""))
        var result = int_val / 10 ** (len(string) - pos - 1)
        return Optional(result)
    except:
        return None


fn cast_to_int(string: String) -> Optional[Int]:
    from builtin.string import atol

    try:
        return Optional(atol(string))
    except:
        return None


# fn read_config(file: String) raises -> Tuple[Config, TransformerWeights]:
#     var conf: Config
#     var weights: TransformerWeights
#     with open(file, 'rb') as f:
#         conf = f
#         weights = TransformerWeights(f, conf)

#     return conf, weights


fn main():
    var checkpoint_path: Optional[String]
    var tokenizer_path: String = "tokenizer.bin"
    var temperature = 1.0
    var topp = 0.9
    var steps = 256
    var prompt: Optional[String] = None
    var rng_seed: Optional[Float64] = Optional(0.0)
    var mode: Mode = Mode.Generate
    var system_prompt: Optional[String] = None

    if len(argv()) < 2:
        print("Error: You must have arguments")
        error_usage()
        return

    if "." not in argv()[1]:
        print("Error: Checkpoint path must have an extension. Given:", argv()[1])
        error_usage()
        return

    checkpoint_path = Optional(str(argv()[1]))

    for i in range(2, len(argv()), 2):
        if (i + 1) >= len(argv()):
            print("Error: You should have a value for the flag.")
            error_usage()
            return

        if argv()[i][0] != "-":
            print("Error: Flags must start with a dash. Given:", argv()[i])
            error_usage()
            return

        if len(argv()[i]) != 2:
            print("Error: Flags must be one character long.")
            error_usage()
            return

        if argv()[i][1] == "t":
            var parsed_temp = cast_to_float(str(argv()[i + 1]))
            if not parsed_temp:
                print("Error: Temperature must be a float.")
                error_usage()
                return
            temperature = parsed_temp.value()

        elif argv()[i][1] == "p":
            var parsed_topp = cast_to_float(str(argv()[i + 1]))
            if not parsed_topp:
                print("Error: Top-p must be a float.")
                error_usage()
                return
            topp = parsed_topp.value()

        elif argv()[i][1] == "s":
            var parsed_seed = cast_to_float(str(argv()[i + 1]))
            if not parsed_seed:
                print("Error: Seed must be an integer.")
                error_usage()
                return
            rng_seed = parsed_seed.value()

        elif argv()[i][1] == "n":
            var parsed_steps = cast_to_int(str(argv()[i + 1]))
            if not parsed_steps:
                print("Error: Steps must be an integer.")
                error_usage()
                return
            steps = parsed_steps.value()

        elif argv()[i][1] == "i":
            prompt = Optional(str(argv()[i + 1]))

        elif argv()[i][1] == "z":
            tokenizer_path = argv()[i + 1]

        elif argv()[i][1] == "m":
            if argv()[i + 1] == "generate":
                mode = Mode.Generate
            elif argv()[i + 1] == "chat":
                mode = Mode.Chat
            else:
                print("Error: Mode must be either 'generate' or 'chat'.")
                error_usage()
                return

        elif argv()[i] == "y":
            system_prompt = Optional(str(argv()[i + 1]))

        else:
            print("Error: Unknown flag:", argv()[i])
            print("Available flags: t, p, s, n, i, z, m")
            error_usage()
            return

        # Parameter validation / Overrides

        rng_seed = None if rng_seed and rng_seed.value() <= 0 else rng_seed
        temperature = 0.0 if temperature < 0 else temperature
        topp = 0.9 if 1.0 < topp or topp < 0.0 else topp
        steps = 0 if steps < 0 else steps

        # Build the transformer via model .bin file
        var model_path: String
        if checkpoint_path:
            model_path = checkpoint_path.value()
        else:
            print("Error: Model path was provided.")
            return

        var config: Config
        var weights: TransformerWeights
        try:
            with open(model_path, "rb") as f:
                config = f
                weights = TransformerWeights(f, config)
        except:
            print("Error: Not able to read the model .bin file.", model_path)
            return

        # Build the tokenizer via tokenizer .bin file
        var tokenizer: Tokenizer

        # Build the sampler

        # Run

        # Memory and file handles cleanup
