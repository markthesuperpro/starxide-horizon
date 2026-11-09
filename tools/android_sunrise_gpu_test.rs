// Compare the optimized sunrise-sector shader with the full triangle search on a host GPU.
// The compute module is generated from the production terrain.wgsl by test_sunrise_gpu.py.
#[test]
fn sunrise_sectors_match_reference_on_real_gpu() {
    use super::sunrise_fan;
    use wgpu::util::DeviceExt;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("sunrise shader parity requires a host Vulkan or DirectX 12 GPU adapter");
    eprintln!("Sunrise GPU equivalence: {}", adapter.get_info().name);
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default()))
        .expect("could not create a device for the sunrise shader parity test");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sunrise reference/optimized equivalence"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/sunrise.wgsl")).into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sunrise equivalence"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // 360 yaw angles × 179 pitches, plus the four near-axis directions.
    let mut rays = Vec::<[f32; 4]>::new();
    for yaw in 0..360 {
        for pitch in -89..90 {
            let (sy, cy) = (yaw as f32).to_radians().sin_cos();
            let (sp, cp) = (pitch as f32).to_radians().sin_cos();
            rays.push([sy * cp, sp, cy * cp, 0.0]);
        }
    }
    rays.extend([
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
    ]);

    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sunrise ray directions"),
        contents: bytemuck::cast_slice(&rays),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let result_size = (rays.len() * std::mem::size_of::<[f32; 2]>()) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sunrise weights"),
        size: result_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sunrise weights readback"),
        size: result_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Vary the fan's tilt and the horizon it faces. The full and optimized
    // functions receive the exact same per-frame normals and ray directions.
    for alpha in [0.01, 0.2, 0.65, 1.0] {
        for hemisphere in [-1.0, 1.0] {
            let mut globals = vec![[1.0, 1.0, 1.0, alpha]];
            globals.extend(sunrise_fan::normals(
                alpha,
                glam::Vec3::new(0.0, 0.0, hemisphere),
            ));
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sunrise fan uniforms"),
                contents: bytemuck::cast_slice(&globals),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bindings = [&uniform, &input, &output];
            let entries = bindings
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>();
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sunrise equivalence bindings"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });

            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups((rays.len() as u32).div_ceil(64), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, result_size);
            queue.submit([encoder.finish()]);

            let (tx, rx) = std::sync::mpsc::channel();
            readback.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU polling failed");
            rx.recv()
                .expect("GPU readback callback did not complete")
                .expect("GPU readback mapping failed");

            {
                let mapped = readback.slice(..).get_mapped_range();
                let pairs: &[[f32; 2]] = bytemuck::cast_slice(&mapped);
                assert_eq!(pairs.len(), rays.len());
                for (index, pair) in pairs.iter().enumerate() {
                    assert!(
                        (pair[0] - pair[1]).abs() < 1e-4,
                        "alpha={alpha}, hemisphere={hemisphere}, ray={:?}: reference={} optimized={}",
                        rays[index],
                        pair[0],
                        pair[1],
                    );
                }
            }
            readback.unmap();
        }
    }
    eprintln!(
        "Compared {} ray directions across 4 alpha values and 2 hemispheres ({} evaluations).",
        rays.len(),
        rays.len() * 8
    );
}
