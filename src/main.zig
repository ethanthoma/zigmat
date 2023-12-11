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

    const n: usize = 1024;
    const m: usize = n;

    var mat = try Matrix.init(allocator, n, m);
    var mat2 = try Matrix.identity(allocator, m, n);

    var count: f32 = 1;
    for (0..n) |i| {
        for (0..m) |j| {
            mat.setValue(i, j, count);
            count += 1;
        }
    }

    var timer = try std.time.Timer.start();
    mat.transpose();
    var duration = timer.read();
    var time = try prettyPrintTime(allocator, duration);

    mat.transpose();

    std.debug.print("Matrix transpose of a {}x{} matrices took {s}.\n", .{ n, n, time });

    mat.transpose();

    timer = try std.time.Timer.start();
    const result = try mat.matmul(&mat2, allocator);
    duration = timer.read();
    time = try prettyPrintTime(allocator, duration);

    std.debug.print("Matrix multiplication of two {}x{} matrices took {s}.\n", .{ n, n, time });

    for (0..n) |i| {
        for (0..m) |j| {
            std.debug.assert(mat.data[i * n + j] == result.data[i * n + j]);
        }
    }
}
