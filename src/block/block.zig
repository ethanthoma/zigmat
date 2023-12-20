const cache_line_size = 64;

pub inline fn block_size(comptime T: type) u32 {
    return cache_line_size / @sizeOf(T);
}

pub inline fn Block(comptime T: type) type {
    return [block_size(T) * block_size(T)]T;
}

pub inline fn BlockInfo(comptime T: type) type {
    return struct {
        data: *Block(T),
        height: usize,
        width: usize,
        index: usize = 0,
    };
}

const Matrix = @import("../matrix.zig").Matrix;

pub inline fn copyBlockToMatrix(comptime T: type, matrix: *Matrix(T), matrix_index: usize, block_info: BlockInfo(T)) void {
    @prefetch(block_info.data, .{});

    const size = block_size(T);

    const n = matrix.cols;

    for (0..block_info.height) |i| {
        const block_start_index = i * size;
        const matrix_start_index = matrix_index + i * n;

        const block_vec: @Vector(size, T) = block_info.data[block_start_index..][0..size].*;

        const matrix_ptr: [size]T = @as([*]T, @ptrCast(&matrix.data[matrix_start_index]))[0..size].*;
        const matrix_vec: @Vector(size, T) = matrix_ptr;

        const result: [size]T = block_vec + matrix_vec;

        @memcpy(matrix.data[matrix_start_index .. matrix_start_index + block_info.width], result[0..block_info.width]);
    }
}
