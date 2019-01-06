extern crate env_logger;
#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;
extern crate gfx_hal as hal;
extern crate winit;

use hal::{
    queue, Adapter, Backend, Capability, Gpu, Compute, Instance, PhysicalDevice, QueueFamily,
};
use winit::{dpi, ControlFlow, Event, EventsLoop, Window, WindowBuilder, WindowEvent};

static WINDOW_NAME: &str = "04_logical_device";

fn main() {
    env_logger::init();
    let mut application = HelloTriangleApplication::init();
    application.run();
    application.clean_up();
}

struct WindowState {
    events_loop: EventsLoop,
    _window: Window,
}

struct HalState {
    _command_queues: Vec<queue::CommandQueue<back::Backend, Compute>>,
    _device: <back::Backend as Backend>::Device,
    _adapter: Adapter<back::Backend>,
    _instance: back::Instance,
}

impl HalState {
    fn clean_up(self) {}
}

struct HelloTriangleApplication {
    hal_state: HalState,
    window_state: WindowState,
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

impl HelloTriangleApplication {
    pub fn init() -> HelloTriangleApplication {
        let window_state = HelloTriangleApplication::init_window();
        let hal_state = HelloTriangleApplication::init_hal();

        HelloTriangleApplication {
            hal_state,
            window_state,
        }
    }

    fn init_window() -> WindowState {
        let events_loop = EventsLoop::new();
        let window_builder = WindowBuilder::new()
            .with_dimensions(dpi::LogicalSize::new(1024., 768.))
            .with_title(WINDOW_NAME.to_string());
        let window = window_builder.build(&events_loop).unwrap();
        WindowState {
            events_loop,
            _window: window,
        }
    }

    fn init_hal() -> HalState {
        let instance = HelloTriangleApplication::create_instance();
        let mut adapter = HelloTriangleApplication::pick_adapter(&instance);
        let (device, command_queues) =
            HelloTriangleApplication::create_device_with_compute_queues(&mut adapter);
        HalState {
            _command_queues: command_queues,
            _device: device,
            _adapter: adapter,
            _instance: instance,
        }
    }

    fn create_instance() -> back::Instance {
        back::Instance::create(WINDOW_NAME, 1)
    }

    fn find_queue_families(adapter: &Adapter<back::Backend>) -> QueueFamilyIds {
        let mut queue_family_ids = QueueFamilyIds::default();

        for queue_family in &adapter.queue_families {
            if queue_family.max_queues() > 0 && queue_family.supports_compute() {
                queue_family_ids.compute_family = Some(queue_family.id());
            }

            if queue_family_ids.is_complete() {
                break;
            }
        }

        queue_family_ids
    }

    fn is_adapter_suitable(adapter: &Adapter<back::Backend>) -> bool {
        HelloTriangleApplication::find_queue_families(adapter).is_complete()
    }

    fn pick_adapter(instance: &back::Instance) -> Adapter<back::Backend> {
        let adapters = instance.enumerate_adapters();
        for adapter in adapters {
            if HelloTriangleApplication::is_adapter_suitable(&adapter) {
                return adapter;
            }
        }
        panic!("No suitable adapter");
    }

    fn create_device_with_compute_queues(
        adapter: &mut Adapter<back::Backend>,
    ) -> (
        <back::Backend as Backend>::Device,
        Vec<queue::CommandQueue<back::Backend, Compute>>,
    ) {
        let family = adapter
            .queue_families
            .iter()
            .find(|family| Compute::supported_by(family.queue_type()) && family.max_queues() > 0)
            .expect("Could not find a queue family supporting compute.");

        // we only want to create a single queue
        let priorities = vec![1.0; 1];
        let families = [(family, priorities.as_slice())];

        let Gpu { device, mut queues } = unsafe {
            adapter
                .physical_device
                .open(&families)
                .expect("Could not create device.")
        };

        let mut queue_group = queues
            .take::<Compute>(family.id())
            .expect("Could not take ownership of relevant queue group.");

        let command_queues: Vec<_> = queue_group.queues.drain(..1).collect();

        (device, command_queues)
    }

    fn main_loop(&mut self) {
        self.window_state
            .events_loop
            .run_forever(|event| match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => ControlFlow::Break,
                _ => ControlFlow::Continue,
            });
    }

    fn run(&mut self) {
        self.main_loop();
    }

    fn clean_up(self) {
        self.hal_state.clean_up();
    }
}
