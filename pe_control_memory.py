from amaranth import *
from pe_control_sys import PEControl


class PEControlMemory(Elaboratable):
    def __init__(
        self,
        mem_snapshot,
        num_bits,
        width,
        g_depth,
        l_depth,
        sys_size_bit,
        signed=True,
    ):
        self.pe_control = PEControl(
            num_bits, width, g_depth, l_depth, sys_size_bit, signed=signed
        )
        self.mem = Memory(width=width, depth=g_depth, init=mem_snapshot)

    def elaborate(self, platform):
        m = Module()
        m.submodules.pe_control = pe_control = self.pe_control
        m.submodules.rdport = rdport = self.mem.read_port()
        m.submodules.wrport = wrport = self.mem.write_port()

        m.d.comb += [
            wrport.data.eq(pe_control.out_w_data),
            wrport.addr.eq(pe_control.out_w_addr),
            wrport.en.eq(pe_control.out_w_en),
            rdport.addr.eq(pe_control.out_r_addr),
            pe_control.in_r_data.eq(rdport.data),
        ]

        return m


# helper for test
from enum import IntEnum
import math


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


if __name__ == "__main__":
    num_bits = 8
    width = 32
    g_depth = 1024
    l_depth = 1024
    sys_size_bit = 2
    signed = True

    # NOTE sys_h and sys_w are 0-based
    sys_h = 1
    sys_w = 1
    fan_in = 2

    cnt_bits = int(round(math.log2(l_depth)))
    mem_snapshot = [
        0xCAFE0000,
        # code area
        make_code(
            OPCODE.SET_M,
            0,
            (sys_h << (cnt_bits + sys_size_bit)) | (sys_w << cnt_bits) | fan_in,
        ),
        make_code(OPCODE.LOAD, LOAD_DEST.A, 7),
        make_code(OPCODE.LOAD, LOAD_DEST.W, 11),
        # make_code(OPCODE.EXEC, FIFO.REUSE << 2, 0),  # reuse
        make_code(OPCODE.EXEC, FIFO.FLOW << 2, 0),  # no reuse
        make_code(OPCODE.STORE, 0, 15),
        0xCAFECAFE,
        # data area
        0x1,  # addr 7, activation, from left
        0x1,
        0x3,
        0x4,
        0x5,  # addr 11, weight, from top
        0x6,
        0x7,
        0x8,
        # 0x05000708,
        # 0x01020004,
        # 0x05060700,
    ]
    dut = PEControlMemory(
        mem_snapshot,
        num_bits,
        width,
        g_depth,
        l_depth,
        sys_size_bit,
        signed=signed,
    )

    from amaranth.sim import Simulator


    def bench():
        data = 0xCAFE0000
        num_cycles = 0
        while data != 0xCAFE0001:
            num_cycles += 1
            yield
            data = yield dut.mem._array._inner[0]
        for i in range(sys_h + 1):
            for j in range(sys_w + 1):
                data = yield dut.mem._array._inner[i * (sys_w + 1) + j + 15]
                print(data)
        print(f"{num_cycles} cycles")

    from pathlib import Path

    p = Path(__file__)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_sync_process(bench)

    with open(p.with_suffix(".vcd"), "w") as f:
        with sim.write_vcd(f):
            sim.run()

    # from amaranth.back import verilog

    # top = PEWithMemory()
    # with open(p.with_suffix(".v"), "w") as f:
    #     f.write(verilog.convert(top, ports=[]))

