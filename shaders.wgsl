
@group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> deltaTime: f32;

fn RHS(time: f32, v: vec2<f32>) -> vec2<f32> {
    // v.xy: position
    // dx/dt = -y
    // dy/dt = x
    return vec2<f32>(-0.2*v.y,0.2*v.x);
}

fn RungeKutta(dt: f32, y: vec2<f32>) -> vec2<f32>{
    var timeLeft : f32 = dt;
    var timeStep : f32 = 0.0001;
    timeStep = min(timeStep, timeLeft);
    var Y : vec2<f32> = y;
    while (timeLeft > 0.0) {
        let k1 : vec2<f32> = RHS(timeStep, Y);
        let k2 : vec2<f32> = RHS(timeStep, Y + timeStep * k1 / 2.0);
        let k3 : vec2<f32> = RHS(timeStep, Y + timeStep * k2 / 2.0);
        let k4 : vec2<f32> = RHS(timeStep, Y + timeStep * k3);
        timeLeft -= timeStep; // assuming ode is time-independent
        timeStep = min(timeStep, timeLeft);
        Y += timeStep / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
    return Y;
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let position = positions[index];

    positions[index] = RungeKutta(deltaTime, position);
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(@location(0) inPosition: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(inPosition, 0.0, 1.0);
    out.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
