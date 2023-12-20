const std = @import("std");
const Allocator = std.mem.Allocator;

const Matrix = @import("../matrix.zig").Matrix;

pub fn full(comptime T: type, allocator: Allocator, rows: usize, cols: usize, value: T) Allocator.Error!Matrix(T) {
    const self = try Matrix(T).init(allocator, rows, cols);

    @memset(self.data, value);

    return self;
}

pub fn zeros(comptime T: type, allocator: Allocator, rows: usize, cols: usize) Allocator.Error!Matrix(T) {
    return full(T, allocator, rows, cols, 0);
}

pub fn ones(comptime T: type, allocator: Allocator, rows: usize, cols: usize) Allocator.Error!Matrix(T) {
    return full(T, allocator, rows, cols, 1);
}

const testing = std.testing;

test "full" {
    const allocator = testing.allocator;

    const m = 8;
    const n = 5;
    const value = 3;

    const mat = try full(f32, allocator, m, n, value);
    defer mat.deinit();

    for (0..m) |i| {
        for (0..n) |j| {
            try testing.expectEqual(mat.data[i * n + j], value);
        }
    }
}
