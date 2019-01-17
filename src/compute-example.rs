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
extern crate winit;
#[macro_use]
extern crate lazy_static;

// thanks to @msiglreith, @omni-viral, @termhn, @aleksijuvani, @grovesNL on gfx-rs/ash gitter!

use hal::{
    buffer, command, memory, pool, pso, queue, Adapter, Backend, Capability, Compute,
    DescriptorPool, Device, Features, Gpu, Instance, PhysicalDevice, QueueFamily,
};
use std::{mem, ptr};

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
        log_elapsed("check_result...");
        application.check_result();
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
//        for family in &adapter.queue_families {
//            log_elapsed(&format!("family: {:?}", family));
//        }
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
        let buffer_length: u32 = 16384;
        let buffer_size: u64 = ((mem::size_of::<i32>() as u32)  * buffer_length) as u64;
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
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0, std430) buffer _4_2
{
    int a[16384];
} input_buff;

layout(binding = 1, std430) buffer _4_3
{
    int a[16384];
} output_buff;

shared int foo;

void main()
{
    output_buff.a[gl_GlobalInvocationID.x] = int(gl_LocalInvocationID.x);
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
        buffer_length: u32,
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
        command_buffer.dispatch([buffer_length / 256, 1, 1]);
        command_buffer.finish();

        command_buffer
    }

    unsafe fn execute_compute(&mut self) {
        let calculation_completed_fence = self.device.create_fence(false).unwrap();
        self.command_queues[0].submit_nosemaphores(
            std::iter::once(&self.command_buffer),
            Some(&calculation_completed_fence),
        );
        self.device
            .wait_for_fence(&calculation_completed_fence, std::u64::MAX)
            .unwrap();
        self.device.destroy_fence(calculation_completed_fence);
    }

    unsafe fn fill_payload(&mut self) {
        for j in 0isize..(self.buffer_size as isize) {
            ptr::copy(&rand::random::<u8>(), self.host_memory.offset(j), 1);
        }
    }

    unsafe fn check_result(&self) {
        let base = self.host_memory.add(self.buffer_size as usize);
        for j in 0..16 {
            let x = ptr::read((base as *const u32).add(j as usize));
            //println!("{}: {}", j, x);
        }
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
