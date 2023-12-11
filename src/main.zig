const std = @import("std");

const Matrix = @import("matrix.zig").Matrix;

fn prettyPrintTime(allocator: std.mem.Allocator, duration: u64) ![]const u8 {
    const sec = @as(f64, @floatFromInt(duration)) / 1_000_000_000;
    const ms = @as(f64, @floatFromInt(duration)) / 1_000_000;
    const us = @as(f64, @floatFromInt(duration)) / 1_000;

    if (sec >= 1) {
        return try std.fmt.allocPrint(allocator, "{:.3} s", .{sec});
    } else if (ms >= 1) {
        return try std.fmt.allocPrint(allocator, "{:.3} ms", .{ms});
    } else if (us >= 1) {
        return try std.fmt.allocPrint(allocator, "{:.3} us", .{us});
    } else {
        return try std.fmt.allocPrint(allocator, "{} ns", .{duration});
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // define mat
    const n: usize = 4;

    var mat = try Matrix.init(allocator, n, n);
    defer mat.deinit();

    var count: f32 = 1;
    for (0..n) |i| {
        for (0..n) |j| {
            mat.setValue(i, j, count);
            count += 1;
        }
    }

    // define mat2
    var mat2 = try Matrix.identity(allocator, n, n);
    defer mat2.deinit();
    for (0..n) |i| {
        for (0..n) |j| {
            mat2.setValue(i, j, 1);
        }
    }

    // time gemm
    var timer = try std.time.Timer.start();
    const result = try mat.matmul(&mat2);
    const duration = timer.read();
    const time = try prettyPrintTime(allocator, duration);
    std.debug.print("Matrix multiplication of two {}x{} matrices took {s}.\n", .{ n, n, time });

    // assert gemm
    const tol = std.math.floatEps(f32) * (n + 1);
    for (0..n) |i| {
        var sum_of_row: f32 = 0;
        for (0..n) |j| {
            sum_of_row += mat.data[i * n + j];
        }
        for (0..n) |j| {
            try std.testing.expectApproxEqRel(sum_of_row, result.data[i * n + j], tol);
        }
    }
}
