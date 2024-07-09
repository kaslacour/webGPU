document.addEventListener('DOMContentLoaded', () => {
    initMathField('dx/dt','mathInput1','-y');
    initMathField('dy/dt','mathInput2','x');
    main();
});

async function main() {
    const { device, context, swapChainFormat } = await initWebGPU();

    const shaderModule = device.createShaderModule({
        code: await fetch('shaders.wgsl').then(res => res.text()),
    });

    const numParticles = 1024;
    const particlePositions = new Float32Array(numParticles * 2); // Position only

    for (let i = 0; i < numParticles; i++) {
        particlePositions[i * 2 + 0] = Math.random() * 3 - 1.5; // x position
        particlePositions[i * 2 + 1] = Math.random() * 3 - 1.5; // y position
    }

    const positionBuffer = device.createBuffer({
        size: particlePositions.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });

    new Float32Array(positionBuffer.getMappedRange()).set(particlePositions);
    positionBuffer.unmap();


    const vertexBuffer = device.createBuffer({
        size: particlePositions.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    const deltaTimeBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(deltaTimeBuffer.getMappedRange())[0] = 0.016;
    deltaTimeBuffer.unmap();


    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'uniform' },
            }
        ],
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: positionBuffer,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: deltaTimeBuffer,
                },
            },
        ],
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
    });

    const computePipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: shaderModule,
            entryPoint: 'update',
        },
    });

    const renderPipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: 'vs_main',
            buffers: [
                {
                    arrayStride: 8,
                    attributes: [
                        {
                            shaderLocation: 0,
                            offset: 0,
                            format: 'float32x2',
                        },
                    ],
                },
            ],
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fs_main',
            targets: [
                {
                    format: swapChainFormat,
                },
            ],
        },
        primitive: {
            topology: 'point-list',
        },
    });

    let lastTimeStamp = 0;

    function updateParticles() {
        const commandEncoder = device.createCommandEncoder();

        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(computePipeline);
        computePass.setBindGroup(0, bindGroup);
        computePass.dispatchWorkgroups(numParticles / 64);
        computePass.end();

        device.queue.submit([commandEncoder.finish()]);

        // Copy updated positions from positionBuffer to vertexBuffer
        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(positionBuffer, 0, vertexBuffer, 0, positionBuffer.size);
        device.queue.submit([copyEncoder.finish()]);
    }

    function renderParticles() {
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
        renderPass.setBindGroup(0, bindGroup);  // Set the bind group for the render pass
        renderPass.draw(numParticles);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
    }

    function frame(timeStamp) {
        const deltaTime = (timeStamp - lastTimeStamp) / 1000.0;
        lastTimeStamp = timeStamp;
        device.queue.writeBuffer(deltaTimeBuffer, 0, new Float32Array([deltaTime]));
        updateParticles();
        renderParticles();
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

async function initMathField(label, mathInput, initInput) {
    const mathField = document.getElementById(mathInput);
    const mathLabel = document.getElementById(label);

    // Set the initial label to a LaTeX expression and render it
    const labelExpression = '\\( \\frac{dx}{dt} = \\)';
    mathLabel.innerHTML = labelExpression;
    MathJax.typesetPromise([mathLabel]);

    // Set an initial value (LaTeX format) for the math input field
    mathField.setValue(initInput);

    mathField.addEventListener('input', async () => {
        const enteredMath = mathField.getValue('latex'); // Get the LaTeX math expression
        const normalMath = latexToNormal(enteredMath);

        const wgslCode = await loadWGSLFile('shaders.wgsl');
        const newMathFunction = createNewMathFunction(normalMath);
        const modifiedWGSLCode = insertNewFunction(wgslCode, newMathFunction);



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

    context.configure({
        device: device,
        format: swapChainFormat,
    });

    return { device, context, swapChainFormat };
}

function latexToNormal(latex) {
    const mathJson = MathLive.latexToMathJson(latex);
    const normal = MathLive.mathJsonToText(mathJson, { output: 'ascii-math' });
    return normal;
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
        return `+
        expr + `;
    }
    `;

}

function insertNewFunction(wgslCode, newFunction) {
    // Insert the new function at the beginning of the WGSL code
    return newFunction + '\n' + wgslCode;
}