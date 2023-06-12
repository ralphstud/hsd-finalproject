from amaranth import *
from pe_stack import PEStack
from enum import IntEnum, Enum
import math
from amaranth.lib.fifo import SyncFIFOBuffered
from functools import reduce


'''
@Author: Kiwoong Kim <helloing0119@naver.com>
@Date: 2023.06.14
@Links: https://github.com/ralphstud/hsd-finalproject/tree/helloing

[PEControl]
It implements matrix-matrix outer product.
basically, it is global buffer memory wrapper that
performs control of memory-pe_control communication.

it contains sys_h * sys_w pe_controls (at least 1)
and performs outer product using them.

'''
# general design

# (sys_w+1) by (sys_h+1) PE control
####################
#          W / B
#      -------------
#     |             |
#  A  |    1 x 1    |  . . . (length of sys_size_bit)
#     |             |
#      -------------
#

# instruction set
# OP     V1               V2
# 4b     4b               24b
# load   to[a|b]          from(gb_addr)
# exec   {1b FIFO flow/reuse, 1b activation function code}
#                         value_for_activation # feed until queue is empty
# store                   to(gb_addr)
# flush
# set_m                   fan_in

# memory layout before start
# 0xCAFE0000  # start of code. tell if input is ready or not
# ...         # instructions, always start at 1
# 0xCAFECAFE  # end of code

# memory layout during & after execution
# 0xCAFENNNN  # NNNN is number of stores done

# load
# load from global buffer to [a|b] local buffer
# global buffer address start from gb_addr
# local buffer address always starts from 0
# size is defined by set_m

# store
# store from PE buffer to global buffer
# global buffer address start from gb_addr
# size is defined by set_m

# exec
# execute for fan_in cycles
# V1 is as follows:
# MSB                              LSB
# 1b    1b                   1b    1b
# nop   FIFO B flow/reuse    nop   activation function code

# activation function code
# 0 for None
# 1 for ReLU

# set_m
# store fan_in in specified register

# states
# INIT
# FETCH
# DECODE  # also handle set_m
# LOAD
# EXEC
# FLUSH
# STORE
class STATECODE(str, Enum):
    INIT    = 'INIT'
    FETCH   = 'FETCH'
    DECODE  = 'DECODE' # also handle set_m
    LOAD    = 'LOAD'
    EXEC    = 'EXEC'
    FLUSH   = 'FLUSH'
    STORE   = 'STORE'

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

class PEControl(Elaboratable):
    def __init__(
        self,
        num_bits,
        width,
        g_depth,
        l_depth,
        sys_size_bit,
        signed=True
    ):        
        self.num_bits = num_bits
        self.acc_bits = width
        self.width = width  # global buffer, local buffer width (bits per line)
        self.g_depth = g_depth  # global buffer depth (number of lines)
        self.l_depth = l_depth  # local buffer depth (number of lines)
        self.signed = signed
        self.sys_size_bit = sys_size_bit # determin size of PE array
        self.next_pc_bits = 8

        # private member to help calculating shape
        self._arr_max = 2**(2*sys_size_bit) # size of PE array
        self._row_max = 2**sys_size_bit

        assert width in [32, 64, 128]
        # BRAM constraint of Virtex-6 devices
        assert l_depth in [1024, 2048, 4096, 8192, 16384]
        assert g_depth in [1024, 2048, 4096, 8192, 16384]
        assert l_depth <= g_depth

        self.gb_addr_width = int(round(math.log2(g_depth)))
        self.cnt_bits = int(round(math.log2(l_depth)))

        assert self.cnt_bits <= 24
        assert self.gb_addr_width >= self.next_pc_bits

        # control signal
        # Each PEStacks share PEControl.in_init & PEControl.in_rst
        # as in_init & in_rst seperately
        self.in_rst = Signal(1, reset_less=True)
        self.in_init = Signal(self.cnt_bits)

        # global buffer interface
        self.in_r_data = Signal(width)
        self.out_r_addr = Signal(self.gb_addr_width)
        self.out_w_en = Signal(1)
        self.out_w_data = Signal(width)
        self.out_w_addr = Signal(self.gb_addr_width)

        # global buffer fused address
        self.addr_io = Signal(self.gb_addr_width)
        self.addr_io_ovf = Signal(1)

        #  PE stack
        self.sys_h = Signal(Shape(sys_size_bit, signed=False))
        self.sys_w = Signal(Shape(sys_size_bit, signed=False))
        self.sys_h_cnt = Signal(Shape(sys_size_bit+1, signed=False))
        self.sys_w_cnt = Signal(Shape(sys_size_bit+1, signed=False))
        
        # PEStack array is 1D array
        # but it will be, acutally, considered as 2D array
        # in_a : connected with sys_h
        # in_b : connected with sys_w
        self.pe_stack_arr = [
            PEStack(num_bits=num_bits, width=width,
                    cnt_bits=self.cnt_bits, signed=signed)
            for _ in range(self._arr_max)
        ]

        # 2**sys_size_bit buffer_a + 2**sys_size_bit buffer_b
        # it will be splited for each PE used
        self.lb_a_arr = [
            SyncFIFOBuffered(width=width, depth=l_depth+1)
            for _ in range(self._row_max)]
        self.lb_b_arr = [
            SyncFIFOBuffered(width=width, depth=l_depth+1)
            for _ in range(self._row_max)]

        # next program counter
        self.next_pc = Signal(self.next_pc_bits, reset=1)
        self.next_pc_ovf = Signal(1)

        self.opcode = Signal(4)

        # split v1 for convenient
        # (use only when opcode == OPCODE.EXEC)
        # 1b            1b           2b
        # v1_fifo_a     v1_fifo_b    v1_act
        self.v1 = Signal(4)
        self.v1_fifo_a = Signal(1)
        self.v1_fifo_b = Signal(1)
        self.v1_act = Signal(2)

        # split v2 for convenient
        # (use only when opcode == OPCODE.STORE)
        # bits      sys_size_bit    sys_size_bit    cnt_bits
        # value     v2_h            v2_w            v2_f
        # affet     in_a / sys_h    in_b / sys_w    fan_in         
        self.v2 = Signal(24)
        self.v2_f = Signal(self.cnt_bits)
        self.v2_h = Signal(self.sys_size_bit)
        self.v2_w = Signal(self.sys_size_bit)

        # control fan in
        self.fan_in = Signal(self.cnt_bits)  # storage, used as init
        self.fan_cnt = Signal(self.cnt_bits)  # count
        self.fan_cnt_ovf = Signal(1)
        self.fan_cnt_next = Signal(self.cnt_bits + 1)

        self.reuse_b = Signal(1)

        # 0xCAFE_NNNN where NNNN is number of stores done
        self.magic_cnt = Signal(32)
        self.magic_cnt_ovf = Signal(1)
        self.magic_init = 0xCAFE0000
        self.magic_end = 0xCAFECAFE

        self.pe_star = Signal(width)

        # register for convenience
        self.cache_mem = Signal(4)

    def elaborate(self, platform):
        m = Module()

        # init wire
        m.d.comb += [
            # fan control
            self.fan_cnt_next.eq(self.fan_cnt + 1),

            # gb memory interface
            self.out_r_addr.eq(self.addr_io),
            self.out_w_addr.eq(self.addr_io),

            # opcode interface
            Cat(self.v2, self.v1, self.opcode).eq(self.in_r_data[:32]),
            ### split v1 for convenient
            ### (use only when opcode == OPCODE.EXEC)
            Cat(self.v1_act, self.v1_fifo_b, self.v1_fifo_a).eq(self.v1),
            ### split v2 for convenient
            ### (use only when opcode == OPCODE.store)
            Cat(self.v2_f, self.v2_w, self.v2_h).eq(
                self.v2[:(self.cnt_bits+2*self.sys_size_bit)]
            ),
        ]

        # init io pipe (FIFO)
        for i in range(0, self._row_max):
            m.submodules += self.lb_a_arr[i]
            m.submodules += self.lb_b_arr[i]
            m.d.comb += [
                self.lb_a_arr[i].w_data.eq(self.in_r_data),
                self.lb_b_arr[i].w_data.eq(
                    Mux(self.reuse_b, self.lb_b_arr[i].r_data, self.in_r_data)),
            ]

        # init PEStacks
        for i in range(0, self._row_max):
            for j in range(0, self._row_max):
                m.submodules += self.pe_stack_arr[i*self._row_max + j]
                m.d.comb += [
                    self.pe_stack_arr[i*self._row_max + j].in_a.eq(self.lb_a_arr[i].r_data),
                    self.pe_stack_arr[i*self._row_max + j].in_b.eq(self.lb_b_arr[j].r_data),
                    self.pe_stack_arr[i*self._row_max + j].in_init.eq(self.in_init),
                    self.pe_stack_arr[i*self._row_max + j].in_rst.eq(self.in_rst),
                ]



        for i in range(0, self._arr_max):
            with m.If(i == self.sys_h_cnt*self._row_max + self.sys_w_cnt):
                m.d.sync += [self.pe_star.eq(self.pe_stack_arr[i].out_d)]

        with m.FSM(reset=STATECODE.INIT):
            with m.State(STATECODE.INIT):
                # wait until global buffer address 0 == 0xCAFE0000
                with m.If(self.in_r_data == self.magic_init):
                    m.next = STATECODE.FETCH
                    m.d.sync += [
                        self.magic_cnt.eq(self.magic_init),
                        self.magic_cnt_ovf.eq(0)
                    ]

            with m.State(STATECODE.FETCH):
                # prevent write
                m.d.sync += self.out_w_en.eq(0)

                # transite to decode when ready
                with m.If(self.addr_io == self.next_pc):
                    m.next = STATECODE.DECODE
                    m.d.sync += [
                        Cat(self.next_pc, self.next_pc_ovf).eq(self.next_pc+1)
                    ]

                # set pc
                with m.Else():
                    m.d.sync += [
                        self.addr_io.eq(self.next_pc),
                        self.addr_io_ovf.eq(self.next_pc_ovf),
                    ]


            with m.State(STATECODE.DECODE):
                # when in_r_data == magic_end == 0xCAFE_CAFE
                # state change from DECODE to INIT
                with m.If(self.in_r_data == self.magic_end):
                    m.next = STATECODE.INIT

                with m.Else():
                    with m.Switch(self.opcode):
                        with m.Case(OPCODE.LOAD):
                            # OP     V1               V2
                            # 4b     4b               24b
                            # load   to[a|b]          from(gb_addr)
                            m.next = STATECODE.LOAD

                            # convert r_mode -> w_mode
                            with m.If(self.v1 == LOAD_DEST.A):
                                m.d.sync += [
                                    self.lb_a_arr[0].w_rdy.eq(1),
                                    self.sys_h_cnt.eq(0),
                                    self.sys_w_cnt.eq(0)
                                ]
                            with m.Else():
                                m.d.sync += [
                                    self.lb_b_arr[0].w_rdy.eq(1),
                                    self.sys_h_cnt.eq(0),
                                    self.sys_w_cnt.eq(0)
                                ]

                            # set addr / cache
                            m.d.sync += [
                                Cat(self.addr_io, self.addr_io_ovf).eq(self.v2),
                                self.fan_cnt.eq(0),
                                self.cache_mem.eq(self.v1),
                            ]

                        with m.Case(OPCODE.EXEC):
                            # OP     V1               V2
                            # 4b     4b               24b
                            # exec   {1b FIFO flush/reuse, 1b activation f code}
                            #                         value_for_activation
                            # 2b                   2b
                            # FIFO flush/reuse     activation function code

                            # FIFO flush/reuse
                            # 0   : flush   1   : reuse
                            # MSB : a       LSB : b

                            # activation function code
                            # 0 for None
                            # 1 for ReLU

                            m.next = STATECODE.EXEC

                            # ignore activation for now
                            # ignore reuse_a for now

                            # ensure fifo is readable
                            with m.Switch(self.v1_fifo_b):
                                with m.Case(FIFO.REUSE):
                                    m.d.sync += [self.reuse_b.eq(1)]
                                    for i in range(0, self._row_max):
                                        m.d.sync += [
                                            self.lb_a_arr[i].r_en.eq(1),
                                            self.lb_b_arr[i].r_en.eq(0),
                                        ]

                                with m.Case(FIFO.FLOW):
                                    m.d.sync += [self.reuse_b.eq(0)]
                                    for i in range(0, self._row_max):
                                        m.d.sync += [
                                            self.lb_a_arr[i].r_en.eq(1),
                                            self.lb_b_arr[i].r_en.eq(1),
                                        ]

                            m.d.sync += [
                                self.in_init.eq(self.fan_in),
                                self.fan_cnt.eq(0),
                            ]
                            for i in range(0, self._row_max):
                                m.d.sync += [
                                    self.lb_a_arr[i].w_en.eq(0),
                                    self.lb_b_arr[i].w_en.eq(0),
                                ]

                        with m.Case(OPCODE.STORE):
                            # OP     V1               V2
                            # 4b     4b               24b
                            # store                   to(gb_addr)
                            m.next = STATECODE.STORE
                            m.d.sync += [
                                Cat(self.addr_io, self.addr_io_ovf).eq(self.v2),
                                self.out_w_en.eq(1),
                                self.out_w_data.eq(self.pe_star),
                                self.sys_h_cnt.eq(0),
                                self.sys_w_cnt.eq(0),
                                Cat(self.magic_cnt, self.magic_cnt_ovf).eq(self.magic_cnt+1),
                            ]

                        with m.Case(OPCODE.FLUSH):
                            m.next = STATECODE.FLUSH
                            for i in range(0, self._row_max):
                                m.d.sync += [
                                    self.lb_b_arr[i].r_en.eq(1),
                                ]

                        with m.Case(OPCODE.SET_M):
                            # OP     V1     V2
                            # 4b     4b     rest    ssb     ssb     cnt_bits
                            # set_m  XX     XX      v2_h    v2_w    v2_f
                            m.d.sync += [
                                self.fan_in.eq(self.v2_f),
                                self.sys_h.eq(self.v2_h),
                                self.sys_w.eq(self.v2_w)
                            ]

                            m.next = STATECODE.FETCH
                        with m.Case():  # default
                            m.next = STATECODE.INIT


            with m.State(STATECODE.LOAD):
                # Load global buffer -> FIFO
                with m.If(self.fan_cnt < self.fan_in):
                    m.d.sync += [
                        self.fan_cnt.eq(self.fan_cnt_next),
                    ]
                    with m.If(self.cache_mem == LOAD_DEST.A):
                        # feed rows FIFO
                        for i in range(0, self._row_max):
                            with m.If(i == self.sys_h_cnt):
                                m.d.sync += [self.lb_a_arr[i].w_en.eq(1)]
                            with m.Else():
                                m.d.sync += [self.lb_a_arr[i].w_en.eq(0)]
                                
                    with m.Else(): # self.cache_mem == LOAD_DEST.W
                        # feed cols FIFO
                        for i in range(0, self._row_max):
                            with m.If(i == self.sys_w_cnt):
                                m.d.sync += [self.lb_b_arr[i].w_en.eq(1)]
                            with m.Else():
                                m.d.sync += [self.lb_b_arr[i].w_en.eq(0)]

                # move row / col
                with m.Elif(self.cache_mem == LOAD_DEST.A): 
                    with m.If(self.sys_h_cnt < self.sys_h):
                        m.d.sync += [
                            self.sys_h_cnt.eq(self.sys_h_cnt+1),
                            self.fan_cnt.eq(0)
                        ]
                    with m.Else():
                        m.d.sync += [
                            self.sys_h_cnt.eq(0),
                            self.sys_w_cnt.eq(0),
                        ]
                        for i in range(0, self._row_max):
                            m.d.sync += [
                                self.lb_a_arr[i].w_en.eq(0),
                                self.lb_b_arr[i].w_en.eq(0),
                            ]
                    m.next = STATECODE.FETCH
                with m.Else(): # self.cache_mem == LOAD_DEST.W
                    with m.If(self.sys_w_cnt < self.sys_w):
                        m.d.sync += [
                            self.sys_w_cnt.eq(self.sys_w_cnt+1),
                            self.fan_cnt.eq(0)
                        ]
                    with m.Else():
                        m.d.sync += [
                            self.sys_h_cnt.eq(0),
                            self.sys_w_cnt.eq(0),
                        ]
                        for i in range(0, self._row_max):
                            m.d.sync += [
                                self.lb_a_arr[i].w_en.eq(0),
                                self.lb_b_arr[i].w_en.eq(0),
                            ]
                    m.next = STATECODE.FETCH


            with m.State(STATECODE.EXEC):
                with m.If(self.fan_cnt_next < self.fan_in):
                    m.d.sync += [
                        self.in_init.eq(0),
                        self.fan_cnt.eq(self.fan_cnt_next),
                    ]
                with m.Else():
                    m.d.sync += [self.in_init.eq(0)]
                    for i in range(0,self._row_max):
                        m.d.sync += [
                            self.lb_a_arr[i].r_en.eq(0),
                            self.lb_b_arr[i].r_en.eq(0),
                        ]

                    m.next = STATECODE.FETCH


            with m.State(STATECODE.STORE):
                with m.If(self.sys_h_cnt <= self.sys_h):
                    with m.If(self.sys_w_cnt >= self.sys_w):
                        m.d.sync += [
                            self.sys_w_cnt.eq(0),
                            self.sys_h_cnt.eq(self.sys_h_cnt+1),
                            self.out_w_data.eq(self.pe_star),
                        ]
                    with m.Else():
                        m.d.sync += [
                            self.sys_w_cnt.eq(self.sys_w_cnt+1),
                            self.out_w_data.eq(self.pe_star),
                        ]
                with m.Else():
                    m.d.sync += [
                        # out_w_en is set 0 when STATECODE == FETCH
                        self.out_w_data.eq(self.magic_cnt),
                    ]
                    m.next = STATECODE.FETCH

            with m.State(STATECODE.FLUSH):
                for i in range(0, self._row_max):
                    with m.If(~self.lb_b_arr[i].r_rdy):
                        m.next = STATECODE.FETCH
                        m.d.sync += [
                            self.lb_b_arr[i].r_en.eq(0)
                        ]
        return m


if __name__ == '__main__':
    width = 32
    unit_width = 8
    signed = True
    g_depth = 1024
    l_depth = 1024

    unit_width_mask = (1 << unit_width) - 1

    dut = PEControl(unit_width, width,
                    g_depth=g_depth, l_depth=l_depth, signed=signed)
    dut = ResetInserter(dut.in_rst)(dut)

    from amaranth.sim import Simulator
    import numpy as np
    from collections import deque

    np.random.seed(43)

    def test_case(dut, in_r_data):
        yield dut.in_r_data.eq(int(in_r_data))
        yield
        w_data = yield dut.out_w_data
        return w_data

    def make_code(a, b=0, c=0):
        return (a << 28) + (b << 24) + c

    # NOTE all operations cover timeframe from DECODE to FETCH
    def initialize():
        # INIT(1) + FETCH(2)
        for _ in range(3):
            yield from test_case(dut, in_r_data=0xCAFE0000)

    def load(to, data_arr, fan_in, start_addr):
        code = make_code(OPCODE.LOAD, to, start_addr)
        # DECODE(1) + LOAD(1, memory read delay)
        for _ in range(2):
            yield from test_case(dut, in_r_data=code)

        # LOAD(fan_in) + FETCH(2)
        for _ in range(fan_in):
            yield from test_case(dut, in_r_data=data_arr.popleft())
        for _ in range(2):
            yield from test_case(dut, in_r_data=0xdeadbeef)

    def set_metadata(fan_in):
        code = make_code(OPCODE.SET_M, 0, fan_in)
        # DECODE(1) + FETCH(2)
        for _ in range(3):
            yield from test_case(dut, in_r_data=code)

    def execute(fan_in, reuse=False):
        v2 = ACTCODE.NONE
        if reuse:
            v2 |= (FIFO.REUSE << 2)
        code = make_code(OPCODE.EXEC, v2, 0)
        # DECODE(1) + EXEC(fan_in) + FETCH(2)
        for _ in range(fan_in + 3):
            yield from test_case(dut, in_r_data=code)

    def flush_lb(fan_in_prev=0):
        code = make_code(OPCODE.FLUSH)
        # DECODE(1) + FLUSH(fan_in_prev) + FETCH(2)
        if fan_in_prev == 0:
            fan_in_prev += 1  # at least one cycle to check if queue is empty``
        for _ in range(fan_in_prev + 3):
            yield from test_case(dut, in_r_data=code)

    def store(start_addr):
        code = make_code(
            OPCODE.STORE, 0, start_addr)

        store_data = []

        # DECODE(1) + STORE(1)
        for _ in range(2):
            w_data = yield from test_case(dut, in_r_data=code)
            if w_data >> (width - 1):
                w_data -= (1 << width)
            store_data.append(w_data)
        # FETCH(2)
        yield from test_case(dut, in_r_data=0xff)
        yield from test_case(dut, in_r_data=0xCAFE0000 + 1)

        return store_data[1:]

    def handle_overflow(d):
        sign_max = 2 ** (unit_width - 1) - 1
        sign_min = -(2 ** (unit_width - 1))
        if d > sign_max:
            d -= (2 ** unit_width)
        elif d < sign_min:
            d += (2 ** unit_width)
        return d

    def validate_v(a_v, w_v):
        psum = 0
        a_arr = []
        w_arr = []
        for i, a in enumerate(a_v):
            for j in range(width // unit_width):
                mask = unit_width_mask << (j * unit_width)
                a = handle_overflow((a_v[i] & mask) >> (j * unit_width))
                w = handle_overflow((w_v[i] & mask) >> (j * unit_width))
                a_arr.append(a)
                w_arr.append(w)
                psum += a * w
        return psum, a_arr, w_arr

    def bench():
        num_runs = 4
        is_first = None

        lb_a = deque()
        lb_b = deque()

        for i in range(num_runs):
            is_first = i % 2 == 0

            if is_first:  # use local buffer
                fan_in = np.random.randint(low=0, high=16) % 4 + 1
                data_w = np.random.randint(
                    low=0, high=2**width, size=[fan_in])
                for d in data_w:
                    lb_b.append(d)

            data_a = np.random.randint(
                low=0, high=2**width, size=[fan_in])
            for d in data_a:
                lb_a.append(d)

            if is_first:
                yield from initialize()
                yield from set_metadata(fan_in)
            yield from load(LOAD_DEST.A, lb_a, fan_in, start_addr=8)
            if is_first:
                yield from load(LOAD_DEST.W, lb_b, fan_in, start_addr=12)
            yield from execute(fan_in, reuse=is_first)

            if not is_first:
                yield from flush_lb(len(lb_b))

            store_data = yield from store(start_addr=16)
            gt_psum, gt_a, gt_w = validate_v(data_a, data_w)

            try:
                assert gt_psum == store_data[0]
            except:
                print(gt_a, gt_w, gt_psum, store_data[0])

            # end of call
            if not is_first:
                for _ in range(1):
                    yield from test_case(dut, in_r_data=0xCAFE0000 + 1)

    from amaranth.sim import Simulator

    from pathlib import Path
    p = Path(__file__)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_sync_process(bench)

    with open(p.with_suffix('.vcd'), 'w') as f:
        with sim.write_vcd(f):
            sim.run()

    from amaranth.back import verilog
    top = PEControl(unit_width, width,
                    g_depth=g_depth, l_depth=l_depth, signed=signed)
    with open(p.with_suffix('.v'), 'w') as f:
        f.write(
            verilog.convert(
                top,
                ports=[top.in_r_data, top.out_r_addr,
                       top.out_w_addr, top.out_w_en, top.out_w_data]))
