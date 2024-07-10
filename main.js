document.addEventListener('DOMContentLoaded', () => {
    initMathField('dx/dt', 'mathInput1', '-y');
    initMathField('dy/dt', 'mathInput2', 'x');
    main();
});

async function main() {
    const { device, context, swapChainFormat } = await initWebGPU();
    let wgslCode = await loadWGSLFile('shaders.wgsl');
    let shaderModule = createShaderModule(device, wgslCode);

    const numParticles = 1024;
    const particlePositions = createParticlePositions(numParticles);

    const positionBuffer = createBuffer(device, particlePositions.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, particlePositions);
    const vertexBuffer = createBuffer(device, particlePositions.byteLength, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
    const deltaTimeBuffer = createBuffer(device, 4, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, new Float32Array([0.016]));

    const bindGroupLayout = createBindGroupLayout(device);
    const bindGroup = createBindGroup(device, bindGroupLayout, positionBuffer, deltaTimeBuffer);
    const pipelineLayout = createPipelineLayout(device, bindGroupLayout);

    let computePipeline = createComputePipeline(device, pipelineLayout, shaderModule);
    let renderPipeline = createRenderPipeline(device, pipelineLayout, shaderModule, swapChainFormat);

    let lastTimeStamp = 0;
    function frame(timeStamp) {
        const deltaTime = (timeStamp - lastTimeStamp) / 1000.0;
        lastTimeStamp = timeStamp;
        device.queue.writeBuffer(deltaTimeBuffer, 0, new Float32Array([deltaTime]));
        updateParticles(device, computePipeline, bindGroup, positionBuffer, vertexBuffer);
        renderParticles(device, context, renderPipeline, vertexBuffer, bindGroup);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);

    window.updateShader = async (newMathFunction) => {
        let wgslCode = await loadWGSLFile('shaders.wgsl');
        wgslCode = replaceFunction(wgslCode, newMathFunction);
        shaderModule = createShaderModule(device, wgslCode);
        computePipeline = createComputePipeline(device, pipelineLayout, shaderModule);
        renderPipeline = createRenderPipeline(device, pipelineLayout, shaderModule, swapChainFormat);
    };
}

function createParticlePositions(numParticles) {
    const positions = new Float32Array(numParticles * 2);
    for (let i = 0; i < numParticles; i++) {
        positions[i * 2] = Math.random() * 3 - 1.5;
        positions[i * 2 + 1] = Math.random() * 3 - 1.5;
    }
    return positions;
}

function createBuffer(device, size, usage, data = null) {
    const buffer = device.createBuffer({ size, usage, mappedAtCreation: !!data });
    if (data) {
        new Float32Array(buffer.getMappedRange()).set(data);
        buffer.unmap();
    }
    return buffer;
}

function createBindGroupLayout(device) {
    return device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
        ],
    });
}

function createBindGroup(device, layout, positionBuffer, deltaTimeBuffer) {
    return device.createBindGroup({
        layout,
        entries: [
            { binding: 0, resource: { buffer: positionBuffer } },
            { binding: 1, resource: { buffer: deltaTimeBuffer } }
        ],
    });
}

function createPipelineLayout(device, bindGroupLayout) {
    return device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
}

function createComputePipeline(device, pipelineLayout, shaderModule) {
    return device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: 'update' },
    });
}

function createRenderPipeline(device, pipelineLayout, shaderModule, swapChainFormat) {
    return device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: 'vs_main',
            buffers: [
                {
                    arrayStride: 8,
                    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }]
                },
            ],
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fs_main',
            targets: [{ format: swapChainFormat }],
        },
        primitive: { topology: 'point-list' },
    });
}

function createShaderModule(device, wgslCode) {
    return device.createShaderModule({ code: wgslCode });
}

function updateParticles(device, computePipeline, bindGroup, positionBuffer, vertexBuffer) {
    const commandEncoder = device.createCommandEncoder();

    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(numParticles / 64);
    computePass.end();

    device.queue.submit([commandEncoder.finish()]);

    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(positionBuffer, 0, vertexBuffer, 0, positionBuffer.size);
    device.queue.submit([copyEncoder.finish()]);
}

function renderParticles(device, context, renderPipeline, vertexBuffer, bindGroup) {
    const commandEncoder = device.createCommandEncoder();

    const textureView = context.getCurrentTexture().createView();
    const renderPassDescriptor = {
        colorAttachments: [
            {
                view: textureView,
                loadOp: 'clear',
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                storeOp: 'store',
            },
        ],
    };

    const renderPass = commandEncoder.beginRenderPass(renderPassDescriptor);
    renderPass.setPipeline(renderPipeline);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(numParticles);
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
}

async function initMathField(label, mathInput, initInput) {
    const mathField = document.getElementById(mathInput);
    const mathLabel = document.getElementById(label);
    const labelExpression = '\\( \\frac{dx}{dt} = \\)';
    mathLabel.innerHTML = labelExpression;
    MathJax.typesetPromise([mathLabel]);

    mathField.setValue(initInput);

    mathField.addEventListener('input', async () => {
        const enteredMath = mathField.getValue('latex');
        const normalMath = latexToNormal(enteredMath);
        const newMathFunction = createNewMathFunction(normalMath);
        await window.updateShader(newMathFunction);
    });
}

async function initWebGPU() {
    if (!navigator.gpu) {
        console.error("WebGPU not supported on this browser");
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    const canvas = document.getElementById("gpuCanvas");
    const context = canvas.getContext("webgpu");
    const swapChainFormat = "bgra8unorm";

    context.configure({ device, format: swapChainFormat });

    return { device, context, swapChainFormat };
}

function latexToNormal(latex) {
    const mathJson = MathLive.latexToMathJson(latex);
    return MathLive.mathJsonToText(mathJson, { output: 'ascii-math' });
}

async function loadWGSLFile(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to load WGSL file: ${response.statusText}`);
    }
    return await response.text();
}

function createNewMathFunction(expr) {
    return `
    fn new_math_function(x: f32, y: f32, t:f32) -> f32 {
        return ${expr};
    }`;
}

function replaceFunction(wgslCode, newFunction) {
    const functionName = 'new_math_function';
    const regex = new RegExp(`fn ${functionName}\\(.*?\\{[\\s\\S]*?\\}`, 'g');
    return wgslCode.replace(regex, newFunction);
}