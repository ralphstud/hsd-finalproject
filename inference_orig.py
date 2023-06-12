import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model
import numpy as np
import lightning as L
from torchmetrics.classification import Accuracy


np.set_printoptions(precision=5)


"""### hyper parameter"""

random_seed = 777

learning_rate = 0.001

batch_size = 4

int_bitwidth = 8
assert int_bitwidth in [4, 8]

dataset = "mnist"
assert dataset in ["mnist", "cifar10"]

checkpoint = "weight/model_"
if int_bitwidth == 4:
    checkpoint += "int4_"
    if dataset == "mnist":
        checkpoint += "0.97229"
    elif dataset == "cifar10":
        checkpoint += "0.47949"
    else:
        raise ValueError(dataset)
elif int_bitwidth == 8:
    checkpoint += "int8_"
    if dataset == "mnist":
        checkpoint += "0.97610"
    elif dataset == "cifar10":
        checkpoint += "0.51200"
    else:
        raise ValueError(dataset)
else:
    raise ValueError(int_bitwidth)

checkpoint += ".safetensors"


use_convlowering = True
use_tiling = True
use_sim = True
skip_sim_linear = True

# hardware parameters
num_bits = 8
width = 32
g_depth = 4096
l_depth = 1024
sys_size_bit = 3
signed = True

# NOTE sys_h and sys_w are 0-based
sys_h = 7
sys_w = 7
max_dotprod = 512  # size of dot product, NOT fan_in
# should be multiple of 4 in case of int8 and width=32
assert max_dotprod % (width // num_bits) == 0
tile_size = (sys_h + 1, sys_w + 1, max_dotprod)


cnt_bits = int(round(np.log2(l_depth)))

"""### dataset and data loader"""

if dataset == "mnist":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_dset = dsets.MNIST(
        root="./",
        train=False,
        transform=transform,
        download=True,
    )
elif dataset == "cifar10":
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    test_dset = dsets.CIFAR10(
        root="./",
        train=False,
        download=True,
        transform=transform_test,
    )
else:
    raise ValueError(dataset)


data_loader_test = torch.utils.data.DataLoader(
    dataset=test_dset, batch_size=batch_size, shuffle=False, drop_last=False
)

"""### STE"""


class roundpass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


roundpass = roundpass.apply

"""### Quantization module"""


class Quantizer(nn.Module):
    def __init__(
        self,
        bitwidth=int_bitwidth,
        always_pos=False,
        use_ste=False,
        is_on=False,
    ):
        super(Quantizer, self).__init__()
        self.alpha_baseline = nn.Parameter(
            torch.tensor(0.0), requires_grad=False
        )
        self.alpha_delta = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.bitwidth = nn.Parameter(
            torch.tensor(bitwidth), requires_grad=False
        )
        self.always_pos = always_pos
        self.use_ste = use_ste
        self.is_on = is_on
        self.is_initialized = False

    def get_alpha(self):
        return F.relu(self.alpha_baseline + self.alpha_delta)

    def get_num_steps(self):
        return 2**self.bitwidth - 2

    def get_stepsize(self):
        num_steps = self.get_num_steps()
        alpha = self.get_alpha()
        return 2 * alpha / num_steps

    def quantize(self, x):
        num_steps = self.get_num_steps()
        stepsize = self.get_stepsize()

        if self.always_pos:
            off = self.get_alpha()
        else:
            off = 0

        if self.use_ste:
            scaled = (x - off) / stepsize
            rounded = roundpass(scaled)
            clamped = torch.clamp(rounded, -num_steps // 2, num_steps // 2)
            q_x = clamped * stepsize + off
        else:
            raise NotImplementedError()
        return q_x

    def forward(self, x):
        if self.is_on:
            x = self.quantize(x)
        return x


"""### callback for quantization modules"""


class QuantizerCallback:
    def quantizer_on(self, model):
        print("Turn on quantizer")
        for m in model.modules():
            if isinstance(m, Quantizer):
                m.is_on = True

    def ste_on(self, model):
        print("Assert STE")
        for m in model.modules():
            if isinstance(m, Quantizer):
                m.use_ste = True


"""### Convolution Lowering"""


def conv2d_lowering_w(w):
    c_out = w.shape[0]
    w = w.reshape([c_out, -1])
    return w


def conv2d_lowering_x(x, kernel_size, padding):
    bs, c_in, h_in, w_in = x.shape
    pad_h, pad_w = padding
    k_h, k_w = kernel_size
    inputs_padded = torch.zeros(bs, c_in, h_in + pad_h * 2, w_in + pad_w * 2)
    inputs_padded[..., pad_h : h_in + pad_h, pad_w : w_in + pad_w] = x

    h_out = h_in + 2 * pad_h - k_h + 1
    w_out = w_in + 2 * pad_w - k_w + 1

    lowered_inputs = torch.zeros(bs, k_h * k_w * c_in, h_out * w_out)
    for b in range(bs):
        for c in range(c_in):
            for i in range(k_h):
                for j in range(k_w):
                    lowered_inputs[
                        b, c * k_h * k_w + i * k_w + j, :
                    ] = inputs_padded[
                        b, c, i : i + h_out, j : j + w_out
                    ].reshape(
                        -1
                    )

    return lowered_inputs, h_out, w_out


"""### unit MM to instruction set for our hardware"""
from enum import IntEnum


# 4-bit OPCode
class OPCODE(IntEnum):
    LOAD = 0
    EXEC = 1
    STORE = 2
    FLUSH = 3
    SET_M = 4


# 4-bit LOAD destination
class LOAD_DEST(IntEnum):
    A = 0
    W = 1


# 1-bit activation code
class ACTCODE(IntEnum):
    NONE = 0
    RELU = 1


# 1-bit FIFO B flow/reuse
class FIFO(IntEnum):
    FLOW = 0
    REUSE = 1


def make_code(a, b=0, c=0):
    return (a << 28) + (b << 24) + c


"""### run simulator"""

from amaranth.sim import Simulator
from pe_control_memory import PEControlMemory


num_cycles = 0


def run_sim_unit(memory, res_addr, tile):
    # initialize Simulator
    dut = PEControlMemory(
        memory,
        num_bits,
        width,
        g_depth,
        l_depth,
        sys_size_bit,
        signed=signed,
    )

    result = np.zeros((len(res_addr), tile[0], tile[1]), dtype=np.int32)

    def bench():
        global num_cycles
        local_num_cycles = 0
        data = 0xCAFE0000
        print(f"{len(res_addr)} runs on this memory")
        while data != 0xCAFE0000 + len(res_addr):
            local_num_cycles += 1
            yield
            data = yield dut.mem._array._inner[0]
        for ri, r_addr in enumerate(res_addr):
            for i in range(tile[0]):
                for j in range(tile[1]):
                    data = yield dut.mem._array._inner[r_addr + i * tile[1] + j]
                    # uint32 to int32
                    if data > 0x7FFF_FFFF:
                        data -= 0x1_0000_0000
                    result[ri, i, j] = data

        num_cycles += local_num_cycles

    # run simulator
    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_sync_process(bench)

    print("running simulation...")
    sim.run()
    print("simulation completed")
    return result


def run_sim(memories, res_addrs, res_ids, tile, results):
    for mi, memory in enumerate(memories):
        result = run_sim_unit(
            memory,
            res_addrs[mi],
            tile,
        )
        for ri, (b, i, j, k) in enumerate(res_ids[mi]):
            for t1 in range(tile[0]):
                for t2 in range(tile[1]):
                    results[b, i, j, k, t1, t2] += result[ri, t1, t2]
    return results


"""### generate instructions for our hardware"""


class MemoryBuilder:
    def __init__(
        self,
        max_lines=g_depth,
        l_depth=l_depth,
        sys_size_bit=sys_size_bit,
        sys_h=sys_h,
        sys_w=sys_w,
        width=width,
        num_bits=num_bits,
    ):
        self.cnt_bits = int(round(np.log2(l_depth)))
        self.sys_size_bit = sys_size_bit
        self.sys_h = sys_h
        self.sys_w = sys_w
        self.len_result = (sys_h + 1) * (sys_w + 1)
        self.num_bits = num_bits
        self.width = width
        self.unit_k = width // num_bits

        self.line_mask = (1 << self.width) - 1

        self.max_lines = max_lines
        self.code = [0xCAFE0000]
        self.data_w = []
        self.data_x = []

        self.updated_lines = None
        self.curr_lines = 2  # 0xCAFE0000, 0xCAFECAFE

        self.result_id_1d = []

        # context
        self.last_fan_in = None

    # check if new code and data fits in memory
    def _check_mem_space(self, x, w=None, fan_in=None):
        len_code = 0

        # space for input and output
        len_data = max(x.reshape(-1).shape[0] // self.unit_k, self.len_result)

        if w is not None:
            len_data += w.reshape(-1).shape[0] // self.unit_k
            len_code += 2  # FLUSH, LOAD_W
            if fan_in != self.last_fan_in:
                len_code += 1  # SET_M
        len_code += 3  # LOAD_X, EXEC, STORE

        self.updated_lines = self.curr_lines + len_code + len_data
        return self.updated_lines <= self.max_lines

    def _append_code(self, code):
        self.code.append(code & self.line_mask)
        return

    def append_instruction(self, x, id, w=None, fan_in=None):
        # w is None implies w is reused
        if not self._check_mem_space(x, w, fan_in):
            # Memory is full --> return context
            return self.last_fan_in

        self.curr_lines = self.updated_lines

        if w is not None:
            if fan_in != self.last_fan_in:
                # TODO SET_M
                set_m = make_code()
                self._append_code(set_m)
                self.last_fan_in = fan_in

            # TODO FLUSH --> LOAD_W
            # NOTE LOAD_W requires address to be set.
            # Set specific address later at `get_mem`
            # TODO store data w (to be used at `get_mem`)

        # TODO LOAD_A --> EXEC --> REUSE
        # TODO store data x (to be used at `get_mem`)

        self.result_id_1d.append(id)

        return None

    # float tensor --> int8 --> int32 array
    def _packing(self, data, is_w=False):
        # assume data is padded to be multiple of unit_k
        if not is_w:
            data = data.T

        masked_data = np.zeros(
            [data.shape[0], data.shape[1] // self.unit_k], dtype=np.int32
        )

        # int8 to int32, packing
        unit_mask = (1 << num_bits) - 1
        for i, row in enumerate(data):
            for j in range(len(row) // self.unit_k):
                int32_packed = torch.zeros([1], dtype=torch.int32)
                for k in range(self.unit_k):
                    int32_packed |= (
                        unit_mask & row[j * self.unit_k + k].int()
                    ) << (k * self.num_bits)
                masked_data[i, j] = int32_packed[0].item()

        return masked_data

    # build memory
    def get_mem(self):
        self._append_code(0xCAFECAFE)
        mem = self.code

        result_addr_1d = []
        data_w_i = 0
        data_x_i = 0
        result_addrs_i = 0

        # traverse code to find address placeholder
        fixed_code_len = len(self.code)
        for i in range(fixed_code_len):
            c = mem[i]
            opcode = (c >> 28) & 0xF
            if opcode == OPCODE.LOAD:
                dest = (c >> 24) & 0xF
                if dest == LOAD_DEST.W:
                    # TODO packing (int8 into int32, use `_packing`)
                    # TODO update `data_w_i`

                    # TODO update address of LOAD_W instruction

                    # append data to memory
                    packed = packed.reshape(-1)
                    for v in packed:
                        mem.append(v)

                elif dest == LOAD_DEST.A:
                    # TODO packing (int8 into int32)
                    # TODO update `data_x_i`

                    # TODO update address of LOAD_A instruction

                    # TODO memo destination of STORE instruction

                    # append data to memory
                    packed = packed.reshape(-1)
                    for v in packed:
                        mem.append(v)

                    # make space(zero padding) for result,
                    # if result is larger than input
                    for i in range(self.len_result - packed.shape[0]):
                        mem.append(0)
                else:
                    raise ValueError(f"invalid load destination {dest}")

            elif opcode == OPCODE.STORE:
                # update address
                mem[i] = make_code(
                    OPCODE.STORE, 0, result_addr_1d[result_addrs_i]
                )
                result_addrs_i += 1

        return mem, result_addr_1d, self.result_id_1d


def compile_mm(
    w,
    x,
    tile=tile_size,
    width=width,
    num_bits=num_bits,
):
    # assume w, x are padded to be multiples of tile
    batch_size = x.shape[0]
    M = w.shape[0]
    K = x.shape[1]
    N = x.shape[2]
    assert K == w.shape[1]
    assert M % tile[0] == 0
    assert N % tile[1] == 0
    unit_k = width // num_bits
    assert K % unit_k == 0

    # MemoryBuilder for all
    memories = []
    result_addr_2d = []
    result_id_2d = []

    mb = MemoryBuilder()
    for b in range(batch_size):
        for i in range(M // tile[0]):
            for k in range(int(np.ceil(K / tile[2]))):
                fan_in = min(tile[2], K - k * tile[2]) // unit_k
                w_tile = w[
                    i * tile[0] : (i + 1) * tile[0],
                    k * tile[2] : (k + 1) * tile[2],
                ]
                for j in range(N // tile[1]):
                    x_tile = x[
                        b,
                        k * tile[2] : (k + 1) * tile[2],
                        j * tile[1] : (j + 1) * tile[1],
                    ]
                    last_fan_in = mb.append_instruction(
                        x_tile,
                        id=(b, i, j, k),
                        w=w_tile if j == 0 else None,
                        fan_in=fan_in,
                    )  # x, id, w=None, fan_in=None
                    # memory overflow
                    if last_fan_in is not None:
                        # store memory
                        memory, result_addr_1d, result_id_1d = mb.get_mem()
                        memories.append(memory)
                        result_addr_2d.append(result_addr_1d)
                        result_id_2d.append(result_id_1d)

                        # TODO reset mb
    # TODO handle trailing instructions

    return memories, result_addr_2d, result_id_2d


"""### Tiling"""


def unit_mm(w, x):
    return w @ x


def tiled_mm(
    w,
    x,
    tile=tile_size,
    use_sim=use_sim,
    num_bits=num_bits,
    width=width,
):
    batch_size = x.shape[0]
    K = x.shape[1]
    N = x.shape[2]
    M = w.shape[0]
    assert K == w.shape[1]

    # zero padding for M, N to be multiples of tile
    unit_k = width // num_bits
    new_M = int(np.ceil(M / tile[0])) * tile[0]
    new_N = int(np.ceil(N / tile[1])) * tile[1]
    new_K = int(np.ceil(K / unit_k)) * unit_k

    w = F.pad(w, (0, new_K - K, 0, new_M - M), "constant", 0)
    x = F.pad(x, (0, new_N - N, 0, new_K - K), "constant", 0)

    accum = torch.zeros([batch_size, new_M, new_N])
    if use_sim:
        memories, result_addr_2d, result_id_2d = compile_mm(w, x)

        # call simulator & get result
        results = np.zeros(
            [
                batch_size,
                new_M // tile[0],
                new_N // tile[1],
                int(np.ceil(new_K / tile[2])),
                tile[0],
                tile[1],
            ],
            dtype=np.int32,
        )
        results = run_sim(memories, result_addr_2d, result_id_2d, tile, results)
        results = torch.tensor(results).reshape([-1, tile[0], tile[1]])
        results = torch.transpose(results, 1, 2)
        print(f"{num_cycles} cycles")

        hw_accum = torch.zeros(
            [
                batch_size,
                new_M // tile[0],
                new_N // tile[1],
                int(np.ceil(new_K / tile[2])),
                tile[0],
                tile[1],
            ],
            dtype=torch.int32,
        )
        ri = 0
        for result_id_1d in result_id_2d:
            for b, i, j, k in result_id_1d:
                for t1 in range(tile[0]):
                    for t2 in range(tile[1]):
                        hw_accum[b, i, j, k, t1, t2] += results[ri, t1, t2]
                ri += 1

    # results to accum
    r_index = 0
    for b in range(batch_size):
        for i in range(new_M // tile[0]):
            for k in range(int(np.ceil(new_K / tile[2]))):
                for j in range(new_N // tile[1]):
                    # debug: gt vs hardware on unit MM
                    if use_sim:
                        gt = unit_mm(
                            w[i * tile[0] : (i + 1) * tile[0], :],
                            x[b, :, j * tile[1] : (j + 1) * tile[1]],
                        )
                        hardware = hw_accum[b, i, j, k]
                        if not np.allclose(gt, hardware):
                            print(b, i, j, k)
                    accum[
                        b,
                        i * tile[0] : (i + 1) * tile[0],
                        j * tile[1] : (j + 1) * tile[1],
                    ] += (
                        hw_accum[b, i, j, k]
                        if use_sim
                        else unit_mm(
                            w[
                                i * tile[0] : (i + 1) * tile[0],
                                k * tile[2] : (k + 1) * tile[2],
                            ],
                            x[
                                b,
                                k * tile[2] : (k + 1) * tile[2],
                                j * tile[1] : (j + 1) * tile[1],
                            ],
                        )
                    )
                    r_index += 1

    # remove zero padding
    accum = accum[:, :M, :N]

    return accum


"""### Quantization aware modules"""


class QConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(QConv2d, self).__init__(*args, **kwargs)
        self.q_w = Quantizer()
        self.q_a = Quantizer(always_pos=True)

    def forward(self, x):
        q_w = self.q_w(self.weight)
        q_x = self.q_a(x)

        if use_convlowering:
            w_step = self.q_w.get_stepsize()
            x_step = self.q_a.get_stepsize()

            q_w_int = (q_w / w_step).round()
            q_x_int = (q_x / x_step).round() - (2 ** (num_bits - 1) - 1)

            q_w_int = conv2d_lowering_w(q_w_int)
            q_x_int, h_out, w_out = conv2d_lowering_x(
                q_x_int, self.kernel_size, self.padding
            )

            signedness_bias = (2 ** (num_bits - 1) - 1) * (
                q_w_int @ torch.ones_like(q_x_int)
            )

            if use_tiling:
                mm = tiled_mm(q_w_int, q_x_int)
            else:
                mm = q_w_int @ q_x_int

            # de-bias
            mm = mm + signedness_bias

            mm = (
                mm.reshape(-1, self.out_channels, h_out, w_out)
                * w_step
                * x_step
            )

            if self.bias is not None:
                mm = mm + self.bias
            return mm

        return F.conv2d(
            q_x,
            q_w,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class QLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(QLinear, self).__init__(*args, **kwargs)
        self.q_w = Quantizer()
        self.q_a = Quantizer(always_pos=True)

    def forward(self, x):
        q_w = self.q_w(self.weight)
        q_x = self.q_a(x)

        if use_tiling and not skip_sim_linear:
            w_step = self.q_w.get_stepsize()
            x_step = self.q_a.get_stepsize()

            q_w_int = (q_w / w_step).round()
            q_x_int = (q_x / x_step).round() - (2 ** (num_bits - 1) - 1)

            q_x_int = q_x_int.reshape([q_x_int.shape[0], q_x_int.shape[1], 1])

            signedness_bias = (2 ** (num_bits - 1) - 1) * (
                q_w_int @ torch.ones_like(q_x_int)
            ).reshape([q_x_int.shape[0], -1])

            mm = tiled_mm(q_w_int, q_x_int)
            mm = mm.reshape([q_x_int.shape[0], -1])
            mm = mm + signedness_bias
            mm = mm * w_step * x_step

            if self.bias is not None:
                mm = mm + self.bias
            print(mm.shape)
            return mm
        return F.linear(q_x, q_w, bias=self.bias)


"""### neural network """


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        in_channel = 1 if dataset == "mnist" else 3
        out_channel = 6 if dataset == "mnist" else 16
        self.layer1 = nn.Sequential(
            QConv2d(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            # output shape 26 x 26 x 6 x 9 = 36504
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        res = 26 if dataset == "mnist" else 30
        flattened = res * res * out_channel
        hidden = 30 if dataset == "mnist" else 256
        self.fc1 = QLinear(flattened, hidden, bias=False)
        self.fc2 = QLinear(hidden, 10, bias=False)
        self.layer2 = nn.Sequential(self.fc1, torch.nn.ReLU(), self.fc2)

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)  # Flatten them for FC
        out = self.layer2(out)
        return out


"""### custom function for evaluation"""


def eval_custom(model, dataloader, accuracy_accum):
    accuracy_accum.reset()
    with torch.no_grad():
        for X, Y in dataloader:
            prediction = model(X)
            accuracy_accum(prediction, Y)
            break  # inference on first batch only

    return accuracy_accum.compute()


"""### overall loop"""

if __name__ == "__main__":
    fabric = L.Fabric(
        accelerator="cpu",
        devices=1,
        precision="bf16-mixed",
        callbacks=[QuantizerCallback()],
    )
    fabric.seed_everything(seed=random_seed)

    model = CNN()
    load_model(model, checkpoint)

    # turn on quantizer and STE
    fabric.call("quantizer_on", model=model)
    fabric.call("ste_on", model=model)

    accuracy_accum = Accuracy(task="multiclass", num_classes=10).to(
        fabric.device
    )
    acc = eval_custom(model, data_loader_test, accuracy_accum)
    print(f"{acc}")

    print(torch.__version__)
    print(L.__version__)
