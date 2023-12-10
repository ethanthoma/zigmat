const std = @import("std");

const MatrixError = error{
    NonMatchingDims,
};

const matrix_block_size = 64;

pub const Matrix = struct {
    rows: usize,
    cols: usize,
    data: []f32,

    pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
        const matrix = Matrix{
            .rows = rows,
            .cols = cols,
            .data = try allocator.alloc(f32, rows * cols),
        };

        for (matrix.data) |*cell| {
            cell.* = 0.0;
        }

        return matrix;
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
        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                self.data[self.rows * i + j] = self.data[self.cols * j + i];
            }
        }

        const temp = self.cols;

        self.cols = self.rows;
        self.rows = temp;
    }

    pub fn matmul(self: *Matrix, other: *Matrix, allocator: std.mem.Allocator) !Matrix {
        if (self.cols != other.rows) {
            return MatrixError.NonMatchingDims;
        }

        var result = try Matrix.init(allocator, self.rows, other.cols);

        try blockGemm(self, other, &result, allocator);

        return result;
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

    fn blockGemm(self: *Matrix, other: *Matrix, result: *Matrix, allocator: std.mem.Allocator) !void {
        const num_m_blocks = (self.rows + matrix_block_size - 1) / matrix_block_size;
        const num_n_blocks = (other.rows + matrix_block_size - 1) / matrix_block_size;
        const num_p_blocks = (other.cols + matrix_block_size - 1) / matrix_block_size;

        const one_block = try allocator.alloc(f32, matrix_block_size * matrix_block_size);
        const two_block = try allocator.alloc(f32, matrix_block_size * matrix_block_size);
        const thr_block = try allocator.alloc(f32, matrix_block_size * matrix_block_size);

        other.transpose();

        for (0..num_m_blocks) |m_idx| {
            for (0..num_p_blocks) |p_idx| {
                for (0..num_n_blocks) |n_idx| {
                    // clear local blocks
                    for (0..matrix_block_size) |i| {
                        for (0..matrix_block_size) |j| {
                            const inner_block_index = i * matrix_block_size + j;

                            one_block[inner_block_index] = 0;
                            two_block[inner_block_index] = 0;
                            thr_block[inner_block_index] = 0;
                        }
                    }

                    // set local blocks
                    const self_block_idx = m_idx * matrix_block_size * self.cols + n_idx * matrix_block_size;
                    const other_block_idx = n_idx * matrix_block_size * other.cols + p_idx * matrix_block_size;
                    const result_block_idx = m_idx * matrix_block_size * result.cols + p_idx * matrix_block_size;

                    const self_block_width = @min(n_idx * matrix_block_size + matrix_block_size, self.cols) - (n_idx * matrix_block_size);
                    const self_block_height = @min(m_idx * matrix_block_size + matrix_block_size, self.rows) - (m_idx * matrix_block_size);

                    for (0..self_block_height) |i| {
                        for (0..self_block_width) |j| {
                            const inner_block_index = i * matrix_block_size + j;
                            const self_idx = self_block_idx + i * self.cols + j;

                            one_block[inner_block_index] = self.data[self_idx];
                        }
                    }

                    const other_block_width = @min(p_idx * matrix_block_size + matrix_block_size, other.cols) - (p_idx * matrix_block_size);
                    const other_block_height = @min(n_idx * matrix_block_size + matrix_block_size, other.rows) - (n_idx * matrix_block_size);

                    for (0..other_block_height) |i| {
                        for (0..other_block_width) |j| {
                            const inner_block_index = i * matrix_block_size + j;
                            const other_idx = other_block_idx + i * other.cols + j;

                            two_block[inner_block_index] = other.data[other_idx];
                        }
                    }

                    // perform calc
                    for (0..matrix_block_size) |i| {
                        for (0..matrix_block_size) |j| {
                            for (0..matrix_block_size) |k| {
                                thr_block[i * matrix_block_size + j] += one_block[i * matrix_block_size + k] * two_block[k * matrix_block_size + j];
                            }
                        }
                    }

                    // copy thr_block to result
                    for (0..matrix_block_size) |i| {
                        for (0..matrix_block_size) |j| {
                            const result_idx = result_block_idx + i * result.cols + j;
                            if (result_idx < result.data.len) {
                                result.data[result_idx] += thr_block[i * matrix_block_size + j];
                            }
                        }
                    }
                }
            }
        }

        other.transpose();
    }

    fn addBlocks(block_one: []f32, block_two: []f32, block_three: []f32) void {
        for (block_three, 0..) |val, index| {
            val = block_one[index] + block_two[index];
        }
    }

    fn subBlocks(block_one: []f32, block_two: []f32, block_three: []f32) void {
        for (block_three, 0..) |val, index| {
            val = block_one[index] - block_two[index];
        }
    }
};