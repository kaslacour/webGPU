document.addEventListener('DOMContentLoaded', () => {
    const mathField1 = document.getElementById('mathInput1');
    const mathLabel1 = document.getElementById('dx/dt');

    // Set the initial label to a LaTeX expression and render it
    const labelExpression1 = '\\( \\frac{dx}{dt} = \\)';
    mathLabel1.innerHTML = labelExpression1;
    MathJax.typesetPromise([mathLabel1]);

    // Set an initial value (LaTeX format) for the math input field
    mathField1.setValue('-y');

    mathField1.addEventListener('input', () => {
        const enteredMath = mathField1.getValue('latex'); // Get the LaTeX math expression
        // renderMath(enteredMath);
    });


    const mathField2 = document.getElementById('mathInput2');
    const mathLabel2 = document.getElementById('dy/dt');

    // Set the initial label to a LaTeX expression and render it
    const labelExpression2 = '\\( \\frac{dy}{dt} = \\)';
    mathLabel2.innerHTML = labelExpression2;
    MathJax.typesetPromise([mathLabel2]);

    // Set an initial value (LaTeX format) for the math input field
    mathField2.setValue('x');

    mathField2.addEventListener('input', () => {
        const enteredMath = mathField2.getValue('latex'); // Get the LaTeX math expression
        // renderMath(enteredMath);
    });




    /* function renderMath(latex) {
        const renderedMath = document.getElementById('renderedMath');
        renderedMath.innerHTML = `\\(${latex}\\)`;
        MathJax.typesetPromise([renderedMath]).catch((err) => console.log(err.message));
    } */

    // Initialize WebGPU and start rendering
    main();
});




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

async function main() {
    const { device, context, swapChainFormat } = await initWebGPU();

    const shaderModule = device.createShaderModule({
        code: await fetch('shaders.wgsl').then(res => res.text()),
    });

    const numParticles = 1024;
    const particlePositions = new Float32Array(numParticles * 2); // Position only
    // const particleVelocities = new Float32Array(numParticles * 2); // Velocity only

    for (let i = 0; i < numParticles; i++) {
        particlePositions[i * 2 + 0] = Math.random() * 3 - 1.5; // x position
        particlePositions[i * 2 + 1] = Math.random() * 3 - 1.5; // y position
        //particleVelocities[i * 2 + 0] = - particlePositions[i * 2 + 1]; // x velocity
        //particleVelocities[i * 2 + 1] = particlePositions[i * 2 + 0]; // y velocity
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