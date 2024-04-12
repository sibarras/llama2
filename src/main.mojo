# /* Inference for Llama-2 Transformer model in pure C */
# Port from C to Mojo


@value
@register_passable
struct Config:
    var dim: Int
    var hidden_dim: Int
    var n_layers: Int
    var n_heads: Int
    var n_kv_heads: Int
    var vocab_size: Int
    var seq_len: Int


@value
struct TransformerWeights:
    var token_embedding_table: TensorF32
    var rms_att_weight: TensorF32
    var rms_ffn_weight: TensorF32
    var wq: TensorF32
    var wk: TensorF32
    var wv: TensorF32
    var wo: TensorF32
    var w1: TensorF32
    var w2: TensorF32
    var w3: TensorF32
    var rms_final_weight: TensorF32
    var wcls: TensorF32


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


@value
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

        # Build the tokenizer via tokenizer .bin file

        # Build the sampler

        # Run

        # Memory and file handles cleanup

