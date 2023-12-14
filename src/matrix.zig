const std = @import("std");

const MatrixError = error{
    NonMatchingDims,
    InvalidShape,
};

const Matrix = @This();

rows: usize,
cols: usize,
data: []f32,
allocator: std.mem.Allocator,

pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
    const data = try allocator.alloc(f32, rows * cols);

    @memset(data, 0.0);

    const matrix = Matrix{
        .rows = rows,
        .cols = cols,
        .data = data,
        .allocator = allocator,
    };

    return matrix;
}

pub fn deinit(self: *const Matrix) void {
    self.allocator.free(self.data);
}

pub fn identity(allocator: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
    var matrix = try Matrix.init(allocator, rows, cols);
    const diag = @max(rows, cols);
    for (0..diag) |i| {
        matrix.data[i * rows + i] = 1;
    }
    return matrix;
}

pub fn setValue(self: *Matrix, row: usize, col: usize, value: f32) void {
    if (row < self.rows and col < self.cols) {
        self.data[(self.cols * row) + col] = value;
    }
}

pub fn reshape(self: *Matrix, rows: usize, cols: usize) !void {
    if (self.rows * self.cols != rows * cols) {
        return MatrixError.InvalidShape;
    }

    self.rows = rows;
    self.cols = cols;
}

pub fn transpose(self: *Matrix) !Matrix {
    const m = self.rows;
    const n = self.cols;

    var result = try Matrix.init(self.allocator, n, m);

    for (0..m) |i| {
        for (0..n) |j| {
            result.data[j * m + i] = self.data[i * n + j];
        }
    }

    return result;
}

test "transpose" {
    const allocator = std.testing.allocator;

    const m: usize = 64;
    const n: usize = 30;

    var initial = try Matrix.init(allocator, m, n);
    defer initial.deinit();

    var expected = try Matrix.init(allocator, n, m);
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

    const actual = try initial.transpose();
    defer actual.deinit();

    // verify
    try std.testing.expectEqual(n, actual.rows);
    try std.testing.expectEqual(m, actual.cols);

    const tol = std.math.floatEps(f32);
    for (0..n) |i| {
        for (0..m) |j| {
            try std.testing.expectApproxEqRel(expected.data[i * m + j], actual.data[i * m + j], tol);
        }
    }
}

const block_size = 8;
const Block = [block_size * block_size]f32;

threadlocal var block_A: Block align(64) = undefined;
threadlocal var block_B: Block align(64) = undefined;
threadlocal var block_C: Block align(64) = undefined;

pub fn matmul(self: *Matrix, other: *Matrix) !Matrix {
    if (self.cols != other.rows) {
        return MatrixError.NonMatchingDims;
    }

    var arena = std.heap.ArenaAllocator.init(self.allocator);
    const allocator = arena.allocator();
    defer arena.deinit();

    const m = self.rows;
    const n = other.rows;
    const p = other.cols;

    var result = try Matrix.init(self.allocator, m, p);

    const num_m_blocks = (m + block_size - 1) / block_size;
    const num_n_blocks = (n + block_size - 1) / block_size;
    const num_p_blocks = (p + block_size - 1) / block_size;

    const thread_pool_options = std.Thread.Pool.Options{ .n_jobs = @intCast(num_m_blocks), .allocator = allocator };
    var thread_pool: std.Thread.Pool = undefined;
    try thread_pool.init(thread_pool_options);
    defer thread_pool.deinit();

    // emulate parallel for
    for (0..num_m_blocks) |m_idx| {
        try thread_pool.spawn(processBlocksRange, .{ self, other, &result, m_idx, num_n_blocks, num_p_blocks });
    }

    return result;
}

inline fn processBlocksRange(matrix_A: *Matrix, matrix_B: *Matrix, result: *Matrix, m_idx: usize, num_n_blocks: usize, num_p_blocks: usize) void {
    const m = matrix_A.rows;
    const n = matrix_B.rows;
    const p = matrix_B.cols;

    const block_m_size = @min(m_idx * block_size + block_size, m) - (m_idx * block_size);

    for (0..num_n_blocks) |n_idx| {
        const block_n_size = @min(n_idx * block_size + block_size, n) - (n_idx * block_size);
        for (0..num_p_blocks) |p_idx| {
            const block_p_size = @min(p_idx * block_size + block_size, p) - (p_idx * block_size);

            const block_a_idx = (m_idx * n + n_idx) * block_size;
            copyWithPaddingToBlockA(matrix_A, block_m_size, block_n_size, block_a_idx);

            const block_b_idx = (n_idx * p + p_idx) * block_size;
            transposeCopyWithPaddingToBlockB(matrix_B, block_n_size, block_p_size, block_b_idx);

            multiplyBlocks();

            const block_c_idx = (m_idx * p + p_idx) * block_size;
            copyBlockToMatrix(result, block_m_size, block_p_size, block_c_idx);
        }
    }
}

inline fn copyBlockToMatrix(matrix: *Matrix, block_height: usize, block_width: usize, block_idx: usize) void {
    const n = matrix.cols;

    for (0..block_height) |i| {
        for (0..block_width) |j| {
            const idx = block_idx + i * n + j;
            matrix.data[idx] += block_C[i * block_size + j];
        }
    }
}

inline fn multiplyBlocks() void {
    inline for (0..block_size) |i| {
        inline for (0..block_size) |j| {
            const vec_A: @Vector(block_size, f32) = block_A[i * block_size ..][0..block_size].*;
            const vec_B: @Vector(block_size, f32) = block_B[j * block_size ..][0..block_size].*;

            const sum_vec = vec_A * vec_B;

            block_C[i * block_size + j] = @reduce(.Add, sum_vec);
        }
    }
}

inline fn copyWithPaddingToBlockA(matrix: *Matrix, block_height: usize, block_width: usize, block_idx: usize) void {
    const n = matrix.cols;

    for (0..block_height) |i| {
        const block_start_index = i * block_size;
        const matrix_start_index = block_idx + i * n;

        var arr: [block_size]f32 = @as([*]f32, @ptrCast((matrix.data)[matrix_start_index..]))[0..block_size].*;
        @memcpy(block_A[block_start_index .. block_start_index + block_size], &arr);
        // zero padding
        @memset(block_A[block_start_index + block_width .. block_start_index + block_size], 0);
    }

    // zero padding
    for (block_height..block_size) |i| {
        const block_start_index = i * block_size;
        @memset(block_A[block_start_index .. block_start_index + block_size], 0);
    }
}

inline fn transposeCopyWithPaddingToBlockB(matrix: *Matrix, block_height: usize, block_width: usize, block_idx: usize) void {
    const n = matrix.cols;

    for (0..block_height) |i| {
        for (0..block_width) |j| {
            const block_start_index = j * block_size;
            const other_idx = block_idx + i * n + j;

            block_B[block_start_index + i] = matrix.data[other_idx];
        }
    }

    // zero padding
    for (0..block_size) |j| {
        const block_start_index = j * block_size;
        @memset(block_A[block_start_index + block_height .. block_start_index + block_size], 0);

        if (j >= block_width) {
            @memset(block_B[block_start_index .. block_start_index + block_height], 0);
        }
    }
}

test "matmul" {
    const allocator = std.testing.allocator;

    const m: usize = 8;
    const n: usize = 9;
    const p: usize = 10;

    // define A: matrix going from 1 to m*n
    var A = try Matrix.init(allocator, m, n);
    defer A.deinit();

    var count: f32 = 1;
    for (0..m) |i| {
        for (0..n) |j| {
            A.setValue(i, j, count);
            count += 1;
        }
    }

    // define B: matrix of 1s
    var B = try Matrix.init(allocator, n, p);
    defer B.deinit();

    for (0..n) |i| {
        for (0..p) |j| {
            B.setValue(i, j, 1);
        }
    }

    const C = try A.matmul(&B);
    defer C.deinit();

    // test
    const base_val: f32 = (n * (n + 1)) / 2;
    const tol = std.math.floatEps(f32) * (n + 1);
    for (0..m) |i| {
        const i_f32: f32 = @floatFromInt(i);
        for (0..p) |j| {
            try std.testing.expectApproxEqRel(base_val + i_f32 * n * n, C.data[i * p + j], tol);
        }
    }
}
