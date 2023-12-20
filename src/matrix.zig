const std = @import("std");

const math = std.math;
const mem = std.mem;
const Allocator = mem.Allocator;
const Thread = std.Thread;

const assert = std.debug.assert;
const debug = std.debug;
const testing = std.testing;

const Error = error{
    NonMatchingDims,
    InvalidShape,
    DifferentTypes,
};

const cache_line_size = 64;

pub usingnamespace @import("init/identity.zig");
pub usingnamespace @import("init/full.zig");

const blk = @import("block/block.zig");

// TODO: fn add(), fn sub(), fn mulScaler(), fn mulRowVector, fn mulColVector
// these can all use blocks
// move the threading logic to the top
// can make it quicker by pre-initing threads or by reducing thread stack size
pub fn Matrix(comptime T: type) type {
    assert(@typeInfo(T) == .Float or @typeInfo(T) == .Int);

    return struct {
        const Self = @This();

        rows: usize,
        cols: usize,
        data: []T,
        allocator: Allocator,

        const block_size = blk.block_size(T);
        const Block = blk.Block(T);
        const BlockInfo = blk.BlockInfo(T);

        threadlocal var block_A: Block align(cache_line_size) = undefined;
        threadlocal var block_B: Block align(cache_line_size) = undefined;
        threadlocal var block_C: Block align(cache_line_size) = undefined;

        pub fn init(allocator: Allocator, rows: usize, cols: usize) Allocator.Error!Self {
            return Self{
                .rows = rows,
                .cols = cols,
                .data = try allocator.alignedAlloc(T, cache_line_size, rows * cols),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: Self) void {
            if (@sizeOf(T) > 0) {
                self.allocator.free(@as([]align(cache_line_size) T, @alignCast(self.data)));
            }
        }

        pub fn setValue(self: *Self, row: usize, col: usize, value: T) void {
            if (row < self.rows and col < self.cols) {
                self.data[(self.cols * row) + col] = value;
            }
        }

        pub fn reshape(self: *Self, rows: usize, cols: usize) Error!void {
            if (self.rows * self.cols != rows * cols) {
                return Error.InvalidShape;
            }

            self.rows = rows;
            self.cols = cols;
        }

        pub fn transpose(allocator: Allocator, self: *Self) Allocator.Error!Self {
            const m = self.rows;
            const n = self.cols;

            var result = try Self.init(allocator, n, m);

            const num_m_blocks = (m + block_size - 1) / block_size;
            const num_n_blocks = (n + block_size - 1) / block_size;

            for (0..num_m_blocks) |m_idx| {
                const block_m_size = @min(m_idx * block_size + block_size, m) - (m_idx * block_size);
                for (0..num_n_blocks) |n_idx| {
                    const block_n_size = @min(n_idx * block_size + block_size, n) - (n_idx * block_size);

                    const from_index = (m_idx * n + n_idx) * block_size;
                    const to_index = (n_idx * m + m_idx) * block_size;

                    zeroPaddedTransposeCopyToBlock(self, from_index, .{ .data = &block_A, .height = block_m_size, .width = block_n_size });

                    blk.copyBlockToMatrix(T, &result, to_index, .{ .data = &block_A, .height = block_n_size, .width = block_m_size });
                }
            }

            return result;
        }

        pub fn matmul(allocator: Allocator, left: *Self, right: *Self) (Error || Allocator.Error || Thread.SpawnError)!Self {
            if (@TypeOf(left) != @TypeOf(right)) {
                return Error.DifferentTypes;
            }

            if (left.cols != right.rows) {
                return Error.NonMatchingDims;
            }

            const m = left.rows;
            const n = right.rows;
            const p = right.cols;

            var to = try Self.init(allocator, m, p);

            const num_m_blocks = (m + block_size - 1) / block_size;
            const num_n_blocks = (n + block_size - 1) / block_size;
            const num_p_blocks = (p + block_size - 1) / block_size;

            // emulate parallel for
            // Todo: move threads to init and use for all fns
            const thread_pool_options = std.Thread.Pool.Options{ .n_jobs = @intCast(num_m_blocks), .allocator = to.allocator };
            var thread_pool: std.Thread.Pool = undefined;
            try thread_pool.init(thread_pool_options);
            defer thread_pool.deinit();

            for (0..num_m_blocks) |m_idx| {
                try thread_pool.spawn(processBlocksRange, .{ left, right, &to, m_idx, num_n_blocks, num_p_blocks });
            }

            return to;
        }

        inline fn processBlocksRange(matrix_A: *Self, matrix_B: *Self, result: *Self, m_idx: usize, num_n_blocks: usize, num_p_blocks: usize) void {
            const m = matrix_A.rows;
            const n = matrix_B.rows;
            const p = matrix_B.cols;

            const block_m_size = @min(m_idx * block_size + block_size, m) - (m_idx * block_size);

            for (0..num_n_blocks) |n_idx| {
                const block_n_size = @min(n_idx * block_size + block_size, n) - (n_idx * block_size);
                const block_A_idx = (m_idx * n + n_idx) * block_size;

                zeroPaddedCopyToBlock(matrix_A, block_A_idx, .{ .data = &block_A, .height = block_m_size, .width = block_n_size });

                for (0..num_p_blocks) |p_idx| {
                    const block_p_size = @min(p_idx * block_size + block_size, p) - (p_idx * block_size);

                    const block_B_idx = (n_idx * p + p_idx) * block_size;
                    zeroPaddedTransposeCopyToBlock(matrix_B, block_B_idx, .{ .data = &block_B, .height = block_n_size, .width = block_p_size });

                    multiplyBlocksIntoBlock(&block_A, &block_B, &block_C);

                    const block_C_idx = (m_idx * p + p_idx) * block_size;
                    blk.copyBlockToMatrix(T, result, block_C_idx, .{ .data = &block_C, .height = block_m_size, .width = block_p_size });
                }
            }
        }

        // does A*B = C, where A is row-major and B is col-major
        inline fn multiplyBlocksIntoBlock(left: *Block, right: *Block, to: *Block) void {
            @prefetch(&left, .{});
            @prefetch(&right, .{});
            @prefetch(&to, .{ .rw = .write });

            inline for (0..block_size) |i| {
                const left_vec: @Vector(block_size, T) = left[i * block_size ..][0..block_size].*;

                inline for (0..block_size) |j| {
                    const right_vec: @Vector(block_size, T) = right[j * block_size ..][0..block_size].*;

                    to[i * block_size + j] = @reduce(.Add, left_vec * right_vec);
                }
            }
        }

        inline fn zeroPaddedCopyToBlock(matrix: *Self, matrix_index: usize, block_info: BlockInfo) void {
            @prefetch(block_info.data, .{ .rw = .write });

            const n = matrix.cols;

            for (0..block_info.height) |i| {
                const block_start_index = i * block_size;
                const matrix_start_index = matrix_index + i * n;

                var arr: [block_size]T = @as([*]T, @ptrCast((matrix.data)[matrix_start_index..]))[0..block_size].*;
                @memcpy(block_info.data[block_start_index .. block_start_index + block_size], &arr);
                // zero padding
                @memset(block_info.data[block_start_index + block_info.width .. block_start_index + block_size], 0);
            }

            // zero padding
            for (block_info.height..block_size) |i| {
                const block_start_index = i * block_size;
                @memset(block_info.data[block_start_index .. block_start_index + block_size], 0);
            }
        }

        inline fn zeroPaddedTransposeCopyToBlock(matrix: *Self, matrix_index: usize, block_info: BlockInfo) void {
            @prefetch(block_info.data, .{ .rw = .write });

            const n = matrix.cols;

            for (0..block_info.height) |i| {
                for (0..block_info.width) |j| {
                    const block_start_index = j * block_size;
                    const other_idx = matrix_index + i * n + j;

                    block_info.data[block_start_index + i] = matrix.data[other_idx];
                }
            }

            // zero padding
            inline for (0..block_size) |j| {
                const block_start_index = j * block_size;
                @memset(block_info.data[block_start_index + block_info.height .. block_start_index + block_size], 0);

                if (j >= block_info.width) {
                    @memset(block_info.data[block_start_index .. block_start_index + block_info.height], 0);
                }
            }
        }

        pub fn print(self: *const Self) void {
            std.debug.print("\n", .{});
            for (0..self.rows) |i| {
                for (0..self.cols) |j| {
                    std.debug.print("{}\t", .{self.data[i * self.cols + j]});
                }
                std.debug.print("\n", .{});
            }
        }
    };
}

const M32 = Matrix(f32);

test "transpose" {
    const allocator = testing.allocator;

    const m: usize = M32.block_size;
    const n: usize = M32.block_size + 1;

    var initial = try M32.init(allocator, m, n);
    defer initial.deinit();

    var expected = try M32.init(allocator, n, m);
    defer expected.deinit();

    var count: f32 = 1;
    for (0..m) |i| {
        for (0..n) |j| {
            initial.setValue(i, j, count);
            count += 1;
        }
    }

    count = 1;
    for (0..m) |j| {
        for (0..n) |i| {
            expected.setValue(i, j, count);
            count += 1;
        }
    }

    const actual = try M32.transpose(allocator, &initial);
    defer actual.deinit();

    // verify
    try testing.expectEqual(n, actual.rows);
    try testing.expectEqual(m, actual.cols);

    const tol = std.math.floatEps(f32);
    for (0..n) |i| {
        for (0..m) |j| {
            try testing.expectApproxEqRel(expected.data[i * m + j], actual.data[i * m + j], tol);
        }
    }
}

test "matmul" {
    const allocator = testing.allocator;

    const m: usize = M32.block_size;
    const n: usize = M32.block_size + 1;
    const p: usize = M32.block_size + 2;

    // define A: matrix going from 1 to m*n
    var A = try M32.init(allocator, m, n);
    defer A.deinit();

    var count: f32 = 1;
    for (0..m) |i| {
        for (0..n) |j| {
            A.setValue(i, j, count);
            count += 1;
        }
    }

    // define B: matrix of 1s
    var B = try M32.init(allocator, n, p);
    defer B.deinit();

    for (0..n) |i| {
        for (0..p) |j| {
            B.setValue(i, j, 1);
        }
    }

    const C = try M32.matmul(allocator, &A, &B);
    defer C.deinit();

    // test
    const base_val: f32 = (n * (n + 1)) / 2;
    const tol = (math.floatEps(f32) * n) * (1 + math.floatEps(f32));
    for (0..m) |i| {
        const i_f32: f32 = @floatFromInt(i);
        for (0..p) |j| {
            try testing.expectApproxEqRel(base_val + i_f32 * n * n, C.data[i * p + j], tol);
        }
    }
}
