const std = @import("std");
const math = std.math;

pub const Neuron = struct {
    weights: []f32,
    bias: f32,

    pub fn init(allocator: std.mem.Allocator, input_size: usize) !Neuron {
        return Neuron{
            .weights = try allocator.alloc(f32, input_size),
            .bias = 0,
        };
    }

    pub fn compute(self: Neuron, inputs: []f32) f32 {
        var sum: f32 = 0.0;
        for (0.., self.weights) |i, weight| {
            sum += weight * inputs[i];
        }

        return sum + self.bias;
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // Example data
    var trainInput = [_]f32{ 0.5, -1.2 };
    const trainOutput = 0.1;

    const inputDim = trainInput.len;
    const outDim = 1;

    // Initialize neurons
    var hidden_neuron = try Neuron.init(allocator, inputDim);
    var output_neuron = try Neuron.init(allocator, outDim);

    // Forward pass through the input neuron
    var hidden_output = [outDim]f32{hidden_neuron.compute(trainInput[0..])};

    // Forward pass through the output neuron
    const output = output_neuron.compute(hidden_output[0..]);

    std.debug.print("Network output: {}\n", .{output});
    std.debug.print("Loss: {}\n", .{output - trainOutput});
}
