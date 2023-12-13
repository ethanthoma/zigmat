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

    for (0..rows * cols) |i| {
        data[i] = 0.0;
    }

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

    const m: usize = 7;
    const n: usize = 6;

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

    // time transpose
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

threadlocal var block_A: Block = undefined;
threadlocal var block_B: Block = undefined;
threadlocal var block_C: Block = undefined;

const max_threads = 100000;

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

    if (num_m_blocks > max_threads) {
        const thread_pool_options = std.Thread.Pool.Options{ .allocator = allocator };
        var thread_pool: std.Thread.Pool = undefined;
        try thread_pool.init(thread_pool_options);
        defer thread_pool.deinit();

        for (0..num_n_blocks) |m_idx| {
            try thread_pool.spawn(from0toNum_p_blocks, .{ self, other, &result, m_idx, num_n_blocks, num_p_blocks });
        }
    } else {
        var threads = try allocator.alloc(std.Thread, num_m_blocks);
        for (0..num_m_blocks) |m_idx| {
            threads[m_idx] = try std.Thread.spawn(.{ .allocator = self.allocator }, from0toNum_p_blocks, .{ self, other, &result, m_idx, num_n_blocks, num_p_blocks });
        }

        for (0..num_m_blocks) |i| {
            threads[i].join();
        }
    }

    return result;
}

fn from0toNum_p_blocks(matrix_A: *Matrix, matrix_B: *Matrix, result: *Matrix, m_idx: usize, num_n_blocks: usize, num_p_blocks: usize) void {
    for (0..num_p_blocks) |p_idx| {
        for (0..num_n_blocks) |n_idx| {
            computeBlock(matrix_A, matrix_B, result, m_idx, p_idx, n_idx);
        }
    }
}

inline fn computeBlock(matrix_A: *Matrix, matrix_B: *Matrix, result: *Matrix, m_idx: usize, p_idx: usize, n_idx: usize) void {
    blockA(matrix_A, m_idx, n_idx);
    blockB(matrix_B, n_idx, p_idx);
    multiplyBlocks();
    copyBlockToMatrix(result, m_idx, p_idx);
}

inline fn copyBlockToMatrix(matrix: *Matrix, block_i: usize, block_j: usize) void {
    const m = matrix.rows;
    const p = matrix.cols;

    const block_height = @min(block_i * block_size + block_size, m) - (block_i * block_size);
    const block_width = @min(block_j * block_size + block_size, p) - (block_j * block_size);
    const block_idx = (block_i * p + block_j) * block_size;

    for (0..block_height) |i| {
        for (0..block_width) |j| {
            const idx = block_idx + i * p + j;
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

inline fn blockA(matrix: *Matrix, block_i: usize, block_j: usize) void {
    const m = matrix.rows;
    const n = matrix.cols;

    const block_height = @min(block_i * block_size + block_size, m) - (block_i * block_size);
    const block_width = @min(block_j * block_size + block_size, n) - (block_j * block_size);
    const block_idx = (block_i * n + block_j) * block_size;

    for (0..block_height) |i| {
        for (0..block_width) |j| {
            const inner_block_index = i * block_size + j;
            const idx = block_idx + i * n + j;

            block_A[inner_block_index] = matrix.data[idx];
        }
        // zero padding
        for (block_width..block_size) |j| {
            block_A[i * block_size + j] = 0;
        }
    }

    // zero padding
    for (block_height..block_size) |i| {
        for (0..block_size) |j| {
            block_A[i * block_size + j] = 0;
        }
    }
}

inline fn blockB(matrix: *Matrix, block_i: usize, block_j: usize) void {
    const m = matrix.rows;
    const n = matrix.cols;

    const block_height = @min(block_i * block_size + block_size, m) - (block_i * block_size);
    const block_width = @min(block_j * block_size + block_size, n) - (block_j * block_size);
    const block_idx = (block_i * n + block_j) * block_size;

    for (0..block_height) |i| {
        for (0..block_width) |j| {
            const inner_block_index = j * block_size + i;
            const other_idx = block_idx + i * n + j;

            block_B[inner_block_index] = matrix.data[other_idx];
        }
        // zero padding
        for (block_width..block_size) |j| {
            block_B[j * block_size + i] = 0;
        }
    }

    // zero padding
    for (block_height..block_size) |i| {
        for (0..block_size) |j| {
            block_B[j * block_size + i] = 0;
        }
    }
}

test "matmul" {
    const allocator = std.testing.allocator;

    const m: usize = 100;
    const n: usize = 200;
    const p: usize = 300;

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

    // timer
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
