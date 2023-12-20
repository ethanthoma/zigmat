const std = @import("std");
const Allocator = std.mem.Allocator;

const zigmat = @import("../matrix.zig");
const Matrix = zigmat.Matrix;

pub fn identity(comptime T: type, allocator: Allocator, rows: usize, cols: usize) Allocator.Error!Matrix(T) {
    var matrix = try zigmat.zeros(T, allocator, rows, cols);

    const diag = @min(rows, cols);
    for (0..diag) |i| {
        matrix.data[i * cols + i] = 1;
    }

    return matrix;
}

const testing = std.testing;

test "identity" {
    const allocator = testing.allocator;

    const m = 8;
    const n = 5;

    const mat = try identity(f32, allocator, m, n);
    defer mat.deinit();

    for (0..m) |i| {
        for (0..n) |j| {
            if (i == j) {
                try testing.expectEqual(mat.data[i * n + j], 1);
            } else {
                try testing.expectEqual(mat.data[i * n + j], 0);
            }
        }
    }
}
