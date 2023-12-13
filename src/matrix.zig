const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}

const MatrixError = error{
    NonMatchingDims,
    InvalidShape,
};

const block_size = 32;
const Block = [block_size * block_size]f32;

pub const Matrix = struct {
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
        var timer = try std.time.Timer.start();

        const actual = try initial.transpose();
        defer actual.deinit();

        const duration = timer.read();

        std.debug.print("\nMatrix transpose of a {}x{} matrix took {} ns.\n", .{ m, n, duration });

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

    pub fn matmul(self: *Matrix, other: *Matrix) !Matrix {
        if (self.cols != other.rows) {
            return MatrixError.NonMatchingDims;
        }

        const m = self.rows;
        const n = other.rows;
        const p = other.cols;

        var result = try Matrix.init(self.allocator, m, p);

        const num_m_blocks = (m + block_size - 1) / block_size;
        const num_n_blocks = (n + block_size - 1) / block_size;
        const num_p_blocks = (p + block_size - 1) / block_size;

        var block_A: [block_size * block_size]f32 = undefined;
        var block_B: [block_size * block_size]f32 = undefined;
        var block_C: [block_size * block_size]f32 = undefined;

        for (0..num_m_blocks) |m_idx| {
            for (0..num_p_blocks) |p_idx| {
                for (0..num_n_blocks) |n_idx| {
                    // set block for A
                    blockA(self, m_idx, n_idx, &block_A);

                    // set block for B
                    blockB(other, n_idx, p_idx, &block_B);

                    // perform calc
                    multiplyBlocks(&block_A, &block_B, &block_C);

                    // copy block_C to result
                    copyBlockToMatrix(&result, m_idx, p_idx, &block_C);
                }
            }
        }

        return result;
    }

    inline fn copyBlockToMatrix(matrix: *Matrix, block_i: usize, block_j: usize, block: *Block) void {
        const m = matrix.rows;
        const p = matrix.cols;

        const block_height = @min(block_i * block_size + block_size, m) - (block_i * block_size);
        const block_width = @min(block_j * block_size + block_size, p) - (block_j * block_size);
        const block_idx = block_i * block_size * p + block_j * block_size;

        for (0..block_height) |i| {
            for (0..block_width) |j| {
                const idx = block_idx + i * p + j;
                matrix.data[idx] += block[i * block_size + j];
            }
        }
    }

    inline fn multiplyBlocks(block_A: *Block, block_B: *Block, block_C: *Block) void {
        for (0..block_size) |i| {
            for (0..block_size) |j| {
                const vec_A: @Vector(block_size, f32) = block_A[i * block_size ..][0..block_size].*;
                const vec_B: @Vector(block_size, f32) = block_B[j * block_size ..][0..block_size].*;

                const sum_vec = vec_A * vec_B;

                block_C[i * block_size + j] = @reduce(.Add, sum_vec);
            }
        }
    }

    inline fn blockA(matrix: *Matrix, block_i: usize, block_j: usize, block: *[block_size * block_size]f32) void {
        const m = matrix.rows;
        const n = matrix.cols;

        const block_height = @min(block_i * block_size + block_size, m) - (block_i * block_size);
        const block_width = @min(block_j * block_size + block_size, n) - (block_j * block_size);
        const block_idx = block_i * block_size * n + block_j * block_size;

        for (0..block_height) |i| {
            for (0..block_width) |j| {
                const inner_block_index = i * block_size + j;
                const idx = block_idx + i * n + j;

                block[inner_block_index] = matrix.data[idx];
            }
            // zero padding
            for (block_width..block_size) |j| {
                block[i * block_size + j] = 0;
            }
        }

        // zero padding
        for (block_height..block_size) |i| {
            for (0..block_size) |j| {
                block[i * block_size + j] = 0;
            }
        }
    }

    inline fn blockB(matrix: *Matrix, block_i: usize, block_j: usize, block: *[block_size * block_size]f32) void {
        const m = matrix.rows;
        const n = matrix.cols;

        const block_height = @min(block_i * block_size + block_size, m) - (block_i * block_size);
        const block_width = @min(block_j * block_size + block_size, n) - (block_j * block_size);
        const block_idx = block_i * block_size * n + block_j * block_size;

        for (0..block_height) |i| {
            for (0..block_width) |j| {
                const inner_block_index = j * block_size + i;
                const other_idx = block_idx + i * n + j;

                block[inner_block_index] = matrix.data[other_idx];
            }
            // zero padding
            for (block_width..block_size) |j| {
                block[j * block_size + i] = 0;
            }
        }

        // zero padding
        for (block_height..block_size) |i| {
            for (0..block_size) |j| {
                block[j * block_size + i] = 0;
            }
        }
    }

    test "matmul" {
        const allocator = std.testing.allocator;

        const m: usize = 50;
        const n: usize = 60;
        const p: usize = 70;

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
        var timer = try std.time.Timer.start();

        const C = try A.matmul(&B);
        defer C.deinit();

        const duration = timer.read();

        std.debug.print("\nMatrix multiplication of a {}x{} matrix and a {}x{} matrix took {} ns.\n", .{ m, n, n, p, duration });

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
};
