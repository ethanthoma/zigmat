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

pub fn main() !void {}

test "thread pool" {
    const thread_pool_options = std.Thread.Pool.Options{ .allocator = std.testing.allocator };
    var thread_pool: std.Thread.Pool = undefined;

    try thread_pool.init(thread_pool_options);
    defer thread_pool.deinit();

    const n_jobs = 1000000;
    for (0..n_jobs) |i| {
        try thread_pool.spawn(threadFunction, .{i});
    }
}

fn threadFunction(arg: usize) void {
    std.debug.print("Thread: {}\n", .{arg});
}
