extern crate env_logger;
#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;
extern crate gfx_hal as hal;
extern crate glsl_to_spirv;
extern crate rand;
extern crate shaderc;
extern crate png;
#[macro_use]
extern crate lazy_static;

// thanks to @msiglreith, @omni-viral, @termhn, @aleksijuvani, @grovesNL on gfx-rs/ash gitter!

use hal::{
    buffer, command, memory, pool, pso, queue, Adapter, Backend, Capability, Compute,
    DescriptorPool, Device, Features, Gpu, Instance, PhysicalDevice, QueueFamily,
};
use std::{mem, ptr};

use std::path::Path;
use std::fs::File;
use std::io::BufWriter;
use png::HasParameters;

lazy_static! {
    static ref START_TIME: std::time::Instant = std::time::Instant::now();
}

fn log_elapsed(text: &str) {
    let elapsed = START_TIME.elapsed();
    let elapsed = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64) * 1e-9;
    println!("{}: {}ms", text, elapsed * 1e3);
}

fn main() {
    log_elapsed("start");
    env_logger::init();
    unsafe {
        log_elapsed("init...");
        let mut application = ComputeApplication::init();
        log_elapsed("fill_payload...");
        application.fill_payload();
        log_elapsed("execute_compute...");
        application.execute_compute();
        log_elapsed("writing png...");
        application.write_png();
        log_elapsed("clean_up...");
        application.clean_up();
        log_elapsed("done...");
    }
}

#[derive(Default)]
struct QueueFamilyIds {
    compute_family: Option<queue::QueueFamilyId>,
}

impl QueueFamilyIds {
    fn is_complete(&self) -> bool {
        self.compute_family.is_some()
    }
}

struct ComputeApplication {
    command_buffer:
    command::CommandBuffer<back::Backend, Compute, command::OneShot, command::Primary>,
    command_pool: pool::CommandPool<back::Backend, Compute>,
    descriptor_pool: <back::Backend as Backend>::DescriptorPool,
    compute_pipeline: <back::Backend as Backend>::ComputePipeline,
    descriptor_set_layout: <back::Backend as Backend>::DescriptorSetLayout,
    pipeline_layout: <back::Backend as Backend>::PipelineLayout,
    host_memory: *mut u8,
    out_buffer: <back::Backend as Backend>::Buffer,
    in_buffer: <back::Backend as Backend>::Buffer,
    memory: <back::Backend as Backend>::Memory,
    buffer_size: u64,
    command_queues: Vec<queue::CommandQueue<back::Backend, Compute>>,
    device: <back::Backend as Backend>::Device,
    _adapter: Adapter<back::Backend>,
    _instance: back::Instance,
}

impl ComputeApplication {
    unsafe fn init() -> ComputeApplication {
        let instance = ComputeApplication::create_instance();
        let mut adapter = ComputeApplication::pick_adapter(&instance);
        log_elapsed("adapter picked");
        let (device, command_queues, queue_type, qf_id) =
            ComputeApplication::create_device_with_compute_queue(&mut adapter);
        log_elapsed("device created, creating io buffers");
        let (buffer_length, buffer_size, memory, in_buffer, out_buffer, host_memory) =
            ComputeApplication::create_io_buffers(&mut adapter, &device);
        let (descriptor_set_layout, pipeline_layout, compute_pipeline) =
            ComputeApplication::create_compute_pipeline(&device);

        let (descriptor_pool, descriptor_sets) = ComputeApplication::set_up_descriptor_sets(
            &device,
            &descriptor_set_layout,
            &in_buffer,
            &out_buffer,
        );

        log_elapsed("creating command pool");
        let mut command_pool = ComputeApplication::create_command_pool(&device, queue_type, qf_id);

        let command_buffer = ComputeApplication::create_command_buffer(
            buffer_length,
            &mut command_pool,
            &descriptor_sets,
            &pipeline_layout,
            &compute_pipeline,
        );

        ComputeApplication {
            command_buffer,
            command_pool,
            descriptor_pool,
            compute_pipeline,
            descriptor_set_layout,
            pipeline_layout,
            host_memory,
            out_buffer,
            in_buffer,
            memory,
            buffer_size,
            command_queues,
            device,
            _adapter: adapter,
            _instance: instance,
        }
    }

    fn create_instance() -> back::Instance {
        log_elapsed("creating instance...");
        let instance = back::Instance::create("compute-example", 1);
        log_elapsed("done");
        instance
    }

    fn find_queue_families(adapter: &Adapter<back::Backend>) -> QueueFamilyIds {
        let mut queue_family_ids = QueueFamilyIds::default();

        for queue_family in &adapter.queue_families {
            if queue_family.max_queues() > 0 && queue_family.supports_graphics() {
                queue_family_ids.compute_family = Some(queue_family.id());
            }

            if queue_family_ids.is_complete() {
                break;
            }
        }

        queue_family_ids
    }

    fn is_adapter_suitable(adapter: &Adapter<back::Backend>) -> bool {
        ComputeApplication::find_queue_families(adapter).is_complete()
    }

    fn pick_adapter(instance: &back::Instance) -> Adapter<back::Backend> {
        /*
        for adapter in instance.enumerate_adapters() {
            println!("adapter: {:?}", adapter.info);
        }
        */
        let mut adapters = instance.enumerate_adapters();
        adapters.drain(..0); // 0 = NVidia 1060, 1 = Intel HD 630
        for adapter in adapters {
            if ComputeApplication::is_adapter_suitable(&adapter) {
                return adapter;
            }
        }
        panic!("No suitable adapter");
    }

    fn create_device_with_compute_queue(
        adapter: &mut Adapter<back::Backend>,
    ) -> (
        <back::Backend as Backend>::Device,
        Vec<queue::CommandQueue<back::Backend, Compute>>,
        queue::QueueType,
        queue::family::QueueFamilyId,
    ) {
        for family in &adapter.queue_families {
            log_elapsed(&format!("family: {:?}", family));
        }
        let family = adapter
            .queue_families
            .iter()
            .find(|family| Compute::supported_by(family.queue_type()) && family.max_queues() > 0)
            .expect("Could not find a queue family supporting graphics.");

        let priorities = vec![1.0; 1];
        let families = [(family, priorities.as_slice())];

        let Gpu { device, mut queues } = unsafe {
            adapter
                .physical_device
                .open(&families, Features::empty())
                .expect("Could not create device.")
        };
        log_elapsed("device opened");

        let mut queue_group = queues
            .take::<Compute>(family.id())
            .expect("Could not take ownership of relevant queue group.");

        let command_queues: Vec<_> = queue_group.queues.drain(..1).collect();

        (device, command_queues, family.queue_type(), family.id())
    }

    unsafe fn create_io_buffers(
        adapter: &mut Adapter<back::Backend>,
        device: &<back::Backend as Backend>::Device,
    ) -> (
        u32,
        u64,
        <back::Backend as Backend>::Memory,
        <back::Backend as Backend>::Buffer,
        <back::Backend as Backend>::Buffer,
        *mut u8,
    ) {
        let buffer_length: u32 = 8_388_608;
        let buffer_size: u64 = ((mem::size_of::<i32>() as u32) * buffer_length) as u64;
        let memory_size: u64 = 2 * buffer_size;

        let mut in_buffer = device
            .create_buffer(buffer_size, buffer::Usage::STORAGE)
            .unwrap();
        let mut out_buffer = device
            .create_buffer(buffer_size, buffer::Usage::STORAGE)
            .unwrap();

        let in_buffer_req = device.get_buffer_requirements(&in_buffer);
        let out_buffer_req = device.get_buffer_requirements(&out_buffer);

        let memory_properties = adapter.physical_device.memory_properties();

        let memory_type_id: hal::MemoryTypeId = memory_properties
            .memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                mem_type
                    .properties
                    .contains(memory::Properties::CPU_VISIBLE | memory::Properties::COHERENT)
                    && in_buffer_req.type_mask & (1 << id) != 0
                    && out_buffer_req.type_mask & (1 << id) != 0
                    && memory_size < memory_properties.memory_heaps[mem_type.heap_index]
            })
            .unwrap()
            .into();

        let memory = device
            .allocate_memory(memory_type_id, memory_size)
            .unwrap();

        device
            .bind_buffer_memory(&memory, 0, &mut in_buffer)
            .unwrap();
        device
            .bind_buffer_memory(&memory, buffer_size, &mut out_buffer)
            .unwrap();

        // you can only ever map device memory to host memory once!
        let host_memory = device
            .map_memory(&memory, (0 as u64)..memory_size)
            .unwrap();

        (
            buffer_length,
            buffer_size,
            memory,
            in_buffer,
            out_buffer,
            host_memory,
        )
    }

    unsafe fn create_compute_pipeline(
        device: &<back::Backend as Backend>::Device,
    ) -> (
        <back::Backend as Backend>::DescriptorSetLayout,
        <back::Backend as Backend>::PipelineLayout,
        <back::Backend as Backend>::ComputePipeline,
    ) {
        let source = "#version 450
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(binding = 0, std430) buffer _4_2
{
    // This buffer should probably be smaller
    uint a[8388608];
} input_buff;

layout(binding = 1, std430) buffer _4_3
{
    uint a[8388608];
} output_buff;

shared uint shbuf[2048];

uint pack(vec4 rgba) {
    vec4 a = floor(0.5 + rgba * 255.0);
    return uint(a.x) | (uint(a.y) << 8) | (uint(a.z) << 16) | (uint(a.w) << 24);
}

vec4 unpack(uint rgba) {
    vec4 a = vec4(rgba & 0xff, (rgba >> 8) & 0xff, (rgba >> 16) & 0xff, (rgba >> 24) & 0xff);
    return a * (1.0 / 255.0);
}

void main()
{
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (gl_LocalInvocationID.x == 0 && gl_LocalInvocationID.y == 0) {
        uint global_ixs = 0;
        uint shared_ix = 0;
        bool running = true;
        while (running) {
            uint op = input_buff.a[global_ix++];
            switch (op) {
                case 0:
                    shbuf[shared_ix++] = op;
                    running = false;
                    break;
                case 1:
                    shbuf[shared_ix++] = op;
                    shbuf[shared_ix++] = input_buff.a[global_ix++]; // cx
                    shbuf[shared_ix++] = input_buff.a[global_ix++]; // cy
                    shbuf[shared_ix++] = input_buff.a[global_ix++]; // r
                    shbuf[shared_ix++] = input_buff.a[global_ix++]; // rgba
                    break;
            }
        }
    }
    barrier();
    float fx = x;
    float fy = y;
    vec4 rgba = vec4(fx / 4096.0, fy / 2048.0, 0.5, 1.0);
    uint shared_ix = 0;
    uint op;
    while ((op = shbuf[shared_ix++]) != 0) {
        switch (op) {
            case 1:
                // filled circle
                float dx = fx - uintBitsToFloat(shbuf[shared_ix++]);
                float dy = fy - uintBitsToFloat(shbuf[shared_ix++]);
                float r = uintBitsToFloat(shbuf[shared_ix++]);
                float a = clamp((r * r - dx * dx - dy * dy) / (2 * r + 1), 0, 1);
                vec4 src = unpack(shbuf[shared_ix++]) * a;
                rgba = src + rgba * (1 - src.w);
                break;
        }
    }
    uint width = 4096; // TODO: make this a parameter
    output_buff.a[y * width + x] = pack(rgba);
}";

        log_elapsed("compiling...");
        let mut compiler = shaderc::Compiler::new().unwrap();
        let compilation_result = compiler
            .compile_into_spirv(
                source,
                shaderc::ShaderKind::Compute,
                "shader.glsl",
                "main",
                None,
            )
            .unwrap();
        log_elapsed(&format!("done: {} bytes", compilation_result.as_binary_u8().len()));

        let shader_module = device
            .create_shader_module(compilation_result.as_binary_u8())
            .expect("Error creating shader module.");

        let (descriptor_set_layout, pipeline_layout, compute_pipeline) = {
            let shader_entry_point = pso::EntryPoint {
                entry: "main",
                module: &shader_module,
                specialization: pso::Specialization {
                    constants: &[],
                    data: &[],
                },
            };

            let descriptor_set_layout_bindings: Vec<pso::DescriptorSetLayoutBinding> = vec![
                pso::DescriptorSetLayoutBinding {
                    binding: 0,
                    ty: pso::DescriptorType::StorageBuffer,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 1,
                    ty: pso::DescriptorType::StorageBuffer,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                },
            ];

            let descriptor_set_layout = device
                .create_descriptor_set_layout(descriptor_set_layout_bindings, &[])
                .unwrap();
            let pipeline_layout = device
                .create_pipeline_layout(vec![&descriptor_set_layout], &[])
                .unwrap();

            let compute_pipeline = {
                let compute_pipeline_desc = pso::ComputePipelineDesc {
                    shader: shader_entry_point,
                    layout: &pipeline_layout,
                    flags: pso::PipelineCreationFlags::empty(),
                    parent: pso::BasePipeline::None,
                };

                device
                    .create_compute_pipeline(&compute_pipeline_desc, None)
                    .unwrap()
            };

            (descriptor_set_layout, pipeline_layout, compute_pipeline)
        };

        device.destroy_shader_module(shader_module);
        (descriptor_set_layout, pipeline_layout, compute_pipeline)
    }

    unsafe fn create_command_pool(
        device: &<back::Backend as Backend>::Device,
        queue_type: queue::QueueType,
        qf_id: queue::family::QueueFamilyId,
    ) -> pool::CommandPool<back::Backend, Compute> {
        let raw_command_pool = device
            .create_command_pool(qf_id, pool::CommandPoolCreateFlags::empty())
            .unwrap();

        // safety check necessary before creating a strongly typed command pool
        assert_eq!(Compute::supported_by(queue_type), true);
        pool::CommandPool::new(raw_command_pool)
    }

    unsafe fn set_up_descriptor_sets(
        device: &<back::Backend as Backend>::Device,
        descriptor_set_layout: &<back::Backend as Backend>::DescriptorSetLayout,
        in_buffer: &<back::Backend as Backend>::Buffer,
        out_buffer: &<back::Backend as Backend>::Buffer,
    ) -> (
        <back::Backend as Backend>::DescriptorPool,
        Vec<<back::Backend as Backend>::DescriptorSet>,
    ) {
        let descriptor_pool_size = pso::DescriptorRangeDesc {
            ty: pso::DescriptorType::StorageBuffer,
            count: 2,
        };

        let mut descriptor_pool = device
            .create_descriptor_pool(1, &[descriptor_pool_size])
            .unwrap();

        let descriptor_set = descriptor_pool.allocate_set(descriptor_set_layout).unwrap();

        let in_descriptor =
            hal::pso::Descriptor::Buffer(in_buffer, None..None);

        // how much of the out_buffer do we want to use? all of it, so None..None for "no range", i.e. everything
        let out_descriptor = hal::pso::Descriptor::Buffer(
            out_buffer,
            None..None,
        );

        // how to know that I should be using Some(descriptor) here, based on docs?
        {
            let in_descriptor_set_write = hal::pso::DescriptorSetWrite {
                set: &descriptor_set,
                binding: 0,
                array_offset: 0,
                descriptors: &[in_descriptor],
            };

            let out_descriptor_set_write = hal::pso::DescriptorSetWrite {
                set: &descriptor_set,
                binding: 1,
                array_offset: 0,
                descriptors: &[out_descriptor],
            };

            device.write_descriptor_sets(vec![in_descriptor_set_write, out_descriptor_set_write]);
        }

        (descriptor_pool, vec![descriptor_set])
    }

    unsafe fn create_command_buffer<'a>(
        _buffer_length: u32,
        command_pool: &'a mut pool::CommandPool<back::Backend, Compute>,
        descriptor_sets: &[<back::Backend as Backend>::DescriptorSet],
        pipeline_layout: &'a <back::Backend as Backend>::PipelineLayout,
        pipeline: &<back::Backend as Backend>::ComputePipeline,
    ) -> command::CommandBuffer<back::Backend, Compute, command::OneShot, command::Primary> {
        let mut command_buffer: command::CommandBuffer<
            back::Backend,
            Compute,
            command::OneShot,
            command::Primary,
        > = command_pool.acquire_command_buffer();

        command_buffer.begin();
        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_compute_descriptor_sets(pipeline_layout, 0, descriptor_sets, &[]);
        command_buffer.dispatch([128, 64, 1]);
        command_buffer.finish();

        command_buffer
    }

    unsafe fn execute_compute(&mut self) {
        log_elapsed("submitting compute");
        let calculation_completed_fence = self.device.create_fence(false).unwrap();
        self.command_queues[0].submit_nosemaphores(
            std::iter::once(&self.command_buffer),
            Some(&calculation_completed_fence),
        );
        log_elapsed("submitted, waiting");
        self.device
            .wait_for_fence(&calculation_completed_fence, std::u64::MAX)
            .unwrap();
        self.device.destroy_fence(calculation_completed_fence);
        log_elapsed("done");
    }

    unsafe fn fill_payload(&mut self) {
        let mut d = DisplayListBuilder::new(self.host_memory as *mut u32);
        for _ in 0..400 {
            let cx = rand::random::<f32>() * 4096.0;
            let cy = rand::random::<f32>() * 2048.0;
            let r = rand::random::<f32>() * 100.0;
            let rgba = rand::random::<u32>() | 0xff000000;
            d.circle(cx, cy, r, rgba);
        }
        d.end();
    }

    unsafe fn write_png(&self) {
        let width = 4096;
        let height = 2048;
        let path = Path::new("out.png");
        let file = File::create(path).unwrap();
        let w = BufWriter::new(file);
        let mut encoder = png::Encoder::new(w, width, height);
        encoder.set(png::ColorType::RGBA);
        let mut writer = encoder.write_header().unwrap();
        let base = self.host_memory.add(self.buffer_size as usize);
        let img_slice = std::slice::from_raw_parts(base, (width * height * 4) as usize);
        writer.write_image_data(img_slice).unwrap();
    }

    unsafe fn clean_up(self) {
        let device = &self.device;

        device.destroy_descriptor_pool(self.descriptor_pool);

        device.destroy_command_pool(self.command_pool.into_raw());

        device.destroy_compute_pipeline(self.compute_pipeline);

        device.destroy_descriptor_set_layout(self.descriptor_set_layout);

        device.destroy_pipeline_layout(self.pipeline_layout);

        device.destroy_buffer(self.out_buffer);

        device.destroy_buffer(self.in_buffer);

        device.free_memory(self.memory);
    }
}

struct DisplayListBuilder {
    buf: *mut u32,
    ix: usize,
}

impl DisplayListBuilder {
    pub fn new(buf: *mut u32) -> DisplayListBuilder {
        DisplayListBuilder { buf, ix: 0 }
    }

    pub unsafe fn write_u32(&mut self, x: u32) {
        ptr::write(self.buf.add(self.ix), x);
        self.ix += 1;
    }

    pub unsafe fn write_f32(&mut self, x: f32) {
        ptr::write(self.buf.add(self.ix) as *mut f32, x);
        self.ix += 1;
    }

    pub unsafe fn end(&mut self) {
        self.write_u32(0);
    }

    pub unsafe fn circle(&mut self, cx: f32, cy: f32, r: f32, rgba: u32) {
        self.write_u32(1);
        self.write_f32(cx);
        self.write_f32(cy);
        self.write_f32(r);
        self.write_u32(rgba);
    }
}
