const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}

const MatrixError = error{
    NonMatchingDims,
};

const block_size = 32;

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
            self.data[self.rows * row + col] = value;
        }
    }

    pub fn transpose(self: *Matrix) void {
        const m = self.rows;
        const n = self.cols;

        const num_m_blocks = (m + block_size - 1) / block_size;
        const num_n_blocks = (n + block_size - 1) / block_size;

        for (0..num_m_blocks) |m_block| {
            const m_block_size = @min((m_block + 1) * block_size, m) - m_block * block_size;
            for (m_block..num_n_blocks - 1) |n_block| {
                // Transpose top right block
                for (0..m_block_size) |i| {
                    for (i..block_size) |j| {
                        const m_index = (m_block * block_size + i) * n + n_block * block_size + j;
                        const n_index = (n_block * block_size + j) * m + m_block * block_size + i;

                        const temp = self.data[m_index];
                        self.data[m_index] = self.data[n_index];
                        self.data[n_index] = temp;
                    }
                }
            }

            // transpose bottom left blocks
            const n_block = num_n_blocks - 1;
            const n_block_size = @min(num_n_blocks * block_size, n) - (num_n_blocks - 1) * block_size;
            for (0..n_block_size) |j| {
                for (j..m_block_size) |i| {
                    const m_index = (m_block * block_size + i) * n + n_block * block_size + j;
                    const n_index = (n_block * block_size + j) * m + m_block * block_size + i;

                    const temp = self.data[m_index];
                    self.data[m_index] = self.data[n_index];
                    self.data[n_index] = temp;
                }
            }
        }

        self.rows = n;
        self.cols = m;
    }

    test "transpose" {
        const allocator = std.testing.allocator;

        const n: usize = 4;
        var actual = try Matrix.init(allocator, n, n);
        defer actual.deinit();

        var expected = try Matrix.init(allocator, n, n);
        defer expected.deinit();

        var count: f32 = 1;
        for (0..n) |i| {
            for (0..n) |j| {
                actual.setValue(i, j, count);
                expected.setValue(j, i, count);
                count += 1;
            }
        }

        // time transpose
        var timer = try std.time.Timer.start();

        actual.transpose();

        const duration = timer.read();

        std.debug.print("\nMatrix transpose of a {}x{} matrices took {} ns.\n", .{ n, n, duration });

        // verify
        const tol = std.math.floatEps(f32);
        for (0..n) |i| {
            for (0..n) |j| {
                try std.testing.expectApproxEqRel(expected.data[i * n + j], actual.data[i * n + j], tol);
            }
        }
    }

    pub fn matmul(self: *Matrix, other: *Matrix) !Matrix {
        if (self.cols != other.rows) {
            return MatrixError.NonMatchingDims;
        }

        var result = try Matrix.init(self.allocator, self.rows, other.cols);

        try blockGemm(self, other, &result);

        return result;
    }

    test matmul {
        const allocator = std.testing.allocator;

        const n: usize = 4;
        var A = try Matrix.init(allocator, n, n);
        defer A.deinit();

        var B = try Matrix.init(allocator, n, n);
        defer B.deinit();

        var expected = try Matrix.init(allocator, n, n);
        defer expected.deinit();

        var base_val: f32 = 0;
        for (0..n) |i| {
            base_val += @floatFromInt(i + 1);
        }

        var count: f32 = 1;
        for (0..n) |i| {
            for (0..n) |j| {
                A.setValue(i, j, count);
                B.setValue(i, j, 1);

                const i_f32: f32 = @floatFromInt(i);

                expected.setValue(i, j, base_val + i_f32 * n * n);

                count += 1;
            }
        }

        // time gemm
        var timer = try std.time.Timer.start();

        const actual: Matrix = try A.matmul(&B);
        defer actual.deinit();

        const duration = timer.read();

        std.debug.print("\nMatrix multiplication of two {}x{} matrices took {} ns.\n", .{ n, n, duration });

        // verify
        const tol = std.math.floatEps(f32) * (n + 1);
        for (0..n) |i| {
            for (0..n) |j| {
                try std.testing.expectApproxEqRel(expected.data[i * n + j], actual.data[i * n + j], tol);
            }
        }
    }

    fn gemm(self: *Matrix, other: *Matrix, result: *Matrix) void {
        other.transpose();

        var sum: f32 = 0;
        for (0..self.rows) |i| {
            for (0..other.cols) |j| {
                sum = 0;
                for (0..other.rows) |k| {
                    sum += self.data[self.rows * i + k] * other.data[other.rows * j + k];
                }
                result.data[self.rows * i + j] = sum;
            }
        }

        other.transpose();
    }

    fn blockGemm(self: *Matrix, other: *Matrix, result: *Matrix) !void {
        const blk_f32 = @Vector(block_size, f32);

        const m = self.rows;
        const n = other.rows;
        const p = other.cols;

        const num_m_blocks = (m + block_size - 1) / block_size;
        const num_n_blocks = (n + block_size - 1) / block_size;
        const num_p_blocks = (p + block_size - 1) / block_size;

        var one_block: [block_size * block_size]f32 = undefined;
        var two_block: [block_size * block_size]f32 = undefined;
        var thr_block: [block_size * block_size]f32 = undefined;

        for (0..num_m_blocks) |m_idx| {
            for (0..num_p_blocks) |p_idx| {
                for (0..num_n_blocks) |n_idx| {
                    // clear local blocks
                    for (0..block_size) |i| {
                        for (0..block_size) |j| {
                            const inner_block_index = i * block_size + j;

                            one_block[inner_block_index] = 0;
                            two_block[inner_block_index] = 0;
                            thr_block[inner_block_index] = 0;
                        }
                    }

                    // set local blocks
                    const self_block_width = @min(n_idx * block_size + block_size, self.cols) - (n_idx * block_size);
                    const self_block_height = @min(m_idx * block_size + block_size, self.rows) - (m_idx * block_size);
                    const self_block_idx = m_idx * block_size * self.cols + n_idx * block_size;

                    for (0..self_block_height) |i| {
                        for (0..self_block_width) |j| {
                            const inner_block_index = i * block_size + j;
                            const self_idx = self_block_idx + i * self.cols + j;

                            one_block[inner_block_index] = self.data[self_idx];
                        }
                    }

                    const other_block_width = @min(n_idx * block_size + block_size, n) - (n_idx * block_size);
                    const other_block_height = @min(p_idx * block_size + block_size, p) - (p_idx * block_size);
                    const other_block_idx = p_idx * block_size * n + n_idx * block_size;

                    for (0..other_block_height) |i| {
                        for (0..other_block_width) |j| {
                            const inner_block_index = i * block_size + j;
                            const other_idx = other_block_idx + i * other.cols + j;

                            two_block[inner_block_index] = other.data[other_idx];
                        }
                    }

                    // perform calc
                    for (0..block_size) |i| {
                        for (0..block_size) |j| {
                            const one_vec: blk_f32 = (&one_block)[i * block_size ..][0..block_size].*;
                            const two_vec: blk_f32 = (&two_block)[j * block_size ..][0..block_size].*;

                            const sum_vec = one_vec * two_vec;

                            thr_block[i * block_size + j] = @reduce(.Add, sum_vec);
                        }
                    }

                    // copy thr_block to result
                    const result_block_width = @min(p_idx * block_size + block_size, n) - (p_idx * block_size);
                    const result_block_height = @min(m_idx * block_size + block_size, m) - (m_idx * block_size);
                    const result_block_idx = m_idx * block_size * n + p_idx * block_size;

                    for (0..result_block_height) |i| {
                        for (0..result_block_width) |j| {
                            const result_idx = result_block_idx + i * result.cols + j;
                            result.data[result_idx] += thr_block[i * block_size + j];
                        }
                    }
                }
            }
        }
    }
};
