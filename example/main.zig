const std = @import("std");

const Matrix = @import("zigmat").Matrix;

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

    const m: usize = 1024;
    const n: usize = m;
    const p: usize = n;

    // define A: matrix going from 1 to m*n
    var A = try Matrix(f32).init(allocator, m, n);
    defer A.deinit();

    var count: f32 = 1;
    for (0..m) |i| {
        for (0..n) |j| {
            A.setValue(i, j, count);
            count += 1;
        }
    }

    // define B: matrix of 1s
    var B = try Matrix(f32).init(allocator, n, p);
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
    const time = try prettyPrintTime(allocator, duration);
    defer allocator.free(time);

    std.debug.print("\nMatrix multiplication of a {}x{} matrix and a {}x{} matrix took {s}.\n", .{ m, n, n, p, time });

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
