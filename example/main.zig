const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;

const zigmat = @import("zigmat");

// runs 100 trials of 1024x1024 matrix multiplication and finds the average time
pub fn main() !void {
    const trials = 100;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const m: usize = 1024;
    const n: usize = m;
    const p: usize = n;

    // define A: matrix going from 1 to m*n
    var A = try zigmat.Matrix(f32).init(allocator, m, n);
    defer A.deinit();

    var count: f32 = 1;
    for (0..m) |i| {
        for (0..n) |j| {
            A.setValue(i, j, count);
            count += 1;
        }
    }

    // define B: matrix of 1s
    var B = try zigmat.ones(f32, allocator, n, p);
    defer B.deinit();

    // run trials
    var sum: u64 = 0;
    for (0..trials) |_| {
        sum += try testMat(allocator, &A, &B);
    }

    // calculate average time
    const duration = sum / trials;
    const time = try prettyPrintTime(allocator, duration);
    defer allocator.free(time);

    std.debug.print("\nMatrix multiplication of two {}x{} matrices took an average of {s} over {} trials.\n", .{ n, n, time, trials });
}

fn testMat(allocator: Allocator, A: *zigmat.Matrix(f32), B: *zigmat.Matrix(f32)) !u64 {
    var timer = try std.time.Timer.start();

    const C = try zigmat.Matrix(f32).matmul(allocator, A, B);
    defer C.deinit();

    const duration = timer.read();

    return duration;
}

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
