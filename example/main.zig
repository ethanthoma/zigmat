const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;

const M32 = @import("zigmat").Matrix(f32);

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

// runs 100 trials of 1024x1024 matrix multiplication and finds the average time
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const m: usize = 1024;
    const n: usize = 1024;
    const p: usize = 1024;

    // define A: matrix going from 1 to m*n
    // TODO: create range based initialization
    var A = try M32.init(allocator, m, n);
    defer A.deinit();

    var count: f32 = 1;
    for (0..m) |i| {
        for (0..n) |j| {
            A.setValue(i, j, count);
            count += 1;
        }
    }

    // define B: matrix of 1s
    var B = try M32.ones(allocator, n, p);
    defer B.deinit();

    const trials = 100;
    var sum: u64 = 0;

    for (0..trials) |_| {
        sum += try testMatMul(allocator, &A, &B);
    }

    const duration = sum / trials;

    const time = try prettyPrintTime(allocator, duration);
    defer allocator.free(time);

    std.debug.print("\nMatrix multiplication of two {}x{} matrices took an average of {s} over {} trials.\n", .{ n, n, time, trials });
}

fn testMatMul(allocator: Allocator, A: *M32, B: *M32) !u64 {
    var timer = try std.time.Timer.start();

    const C = try M32.matmul(allocator, A, B);
    defer C.deinit();

    const duration = timer.read();

    return duration;
}
