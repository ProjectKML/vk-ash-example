use std::{ffi::CString, mem::ManuallyDrop, slice, sync::Arc};

use ash::{
    extensions::khr::{DynamicRendering, Surface, Swapchain, Synchronization2},
    vk, Device, Entry, Instance
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use shaderc::{CompileOptions, Compiler, ShaderKind, SpirvVersion};
use winit::{
    dpi::{PhysicalSize, Size},
    event::{Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::{Window, WindowBuilder}
};

const VERTEX_SOURCE: &str = r#"
#version 460

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) out vec3 color;

void main() {
    if(gl_VertexIndex == 0) {
        gl_Position = vec4(-0.5, 0.5, 0.0, 1.0);
        color = vec3(1.0, 0.0, 1.0);
    } else if(gl_VertexIndex == 1) {
        gl_Position = vec4(0.5, 0.5, 0.0, 1.0);
        color = vec3(0.0, 1.0, 0.0);
    } else {
        gl_Position = vec4(0.0, -0.5, 0.0, 1.0);
        color = vec3(1.0, 1.0, 0.0);
    }
}
"#;

const FRAGMENT_SOURCE: &str = r#"
#version 460

layout(location = 0) in vec3 color;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(color, 1.0);
}
"#;

pub const NUM_FRAMES: usize = 2;

pub struct Frame {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,

    pub present_semaphore: vk::Semaphore,
    pub render_semaphore: vk::Semaphore,

    pub fence: vk::Fence,

    device: Arc<Device>
}

impl Frame {
    pub fn new(device: Arc<Device>) -> Self {
        let command_pool = unsafe { device.create_command_pool(&vk::CommandPoolCreateInfo::default(), None) }.unwrap();
        let command_buffer = unsafe { device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(command_pool).command_buffer_count(1)) }.unwrap()[0];
        let present_semaphore = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }.unwrap();
        let render_semaphore = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }.unwrap();
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None) }.unwrap();

        Self {
            command_pool,
            command_buffer,
            present_semaphore,
            render_semaphore,
            fence,
            device
        }
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);

            self.device.destroy_semaphore(self.render_semaphore, None);
            self.device.destroy_semaphore(self.present_semaphore, None);

            self.device.free_command_buffers(self.command_pool, slice::from_ref(&self.command_buffer));
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

pub struct RenderCtx {
    _entry_loader: Entry,

    instance_loader: Instance,
    surface_loader: Surface,

    surface: vk::SurfaceKHR,

    device_loader: Arc<Device>,
    swapchain_loader: Swapchain,
    dynamic_rendering_loader: DynamicRendering,
    synchronization2_loader: Synchronization2,

    direct_queue: vk::Queue,

    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,

    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    frames: Vec<ManuallyDrop<Frame>>
}

impl RenderCtx {
    pub fn new(window: &Window) -> Self {
        let entry_loader = unsafe { Entry::load() }.unwrap();

        let application_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);

        let instance_layers = [b"VK_LAYER_KHRONOS_validation\0".as_ptr().cast()];

        let mut instance_extensions = vec![];
        ash_window::enumerate_required_extensions(window.raw_display_handle())
            .unwrap()
            .iter()
            .for_each(|e| instance_extensions.push(*e));

        let instance_create_info = vk::InstanceCreateInfo::default()
            .enabled_layer_names(&instance_layers)
            .enabled_extension_names(&instance_extensions)
            .application_info(&application_info);

        let instance_loader = unsafe { entry_loader.create_instance(&instance_create_info, None) }.unwrap();
        let surface_loader = Surface::new(&entry_loader, &instance_loader);

        let surface = unsafe { ash_window::create_surface(&entry_loader, &instance_loader, window.raw_display_handle(), window.raw_window_handle(), None) }.unwrap();

        let physical_devices = unsafe { instance_loader.enumerate_physical_devices() }.unwrap();
        let physical_device = physical_devices[0];

        let mut physical_device_vulkan_12_properties = vk::PhysicalDeviceVulkan12Properties::default();

        let mut physical_device_properties = vk::PhysicalDeviceProperties2::default().push_next(&mut physical_device_vulkan_12_properties);

        unsafe { instance_loader.get_physical_device_properties2(physical_device, &mut physical_device_properties) };

        let queue_priority = 1.0;
        let device_queue_create_info = vk::DeviceQueueCreateInfo::default().queue_priorities(slice::from_ref(&queue_priority));

        let physical_device_features = vk::PhysicalDeviceFeatures::default();

        let mut physical_device_vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default();
        let mut physical_device_dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut physical_device_synchronization_2_features = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);

        let mut physical_device_features = vk::PhysicalDeviceFeatures2::default()
            .features(physical_device_features)
            .push_next(&mut physical_device_vulkan_12_features)
            .push_next(&mut physical_device_dynamic_rendering_features)
            .push_next(&mut physical_device_synchronization_2_features);

        let device_extensions = [Swapchain::name().as_ptr(), DynamicRendering::name().as_ptr(), Synchronization2::name().as_ptr()];

        let device_create_info = vk::DeviceCreateInfo::default()
            .push_next(&mut physical_device_features)
            .queue_create_infos(slice::from_ref(&device_queue_create_info))
            .enabled_extension_names(&device_extensions);

        let device_loader = Arc::new(unsafe { instance_loader.create_device(physical_device, &device_create_info, None) }.unwrap());
        let swapchain_loader = Swapchain::new(&instance_loader, &device_loader);
        let dynamic_rendering_loader = DynamicRendering::new(&instance_loader, &device_loader);
        let synchronization2_loader = Synchronization2::new(&instance_loader, &device_loader);

        let direct_queue = unsafe { device_loader.get_device_queue(0, 0) };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(2)
            .image_format(vk::Format::B8G8R8A8_UNORM)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(vk::Extent2D { width: 1600, height: 900 })
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO);

        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }.unwrap();
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }.unwrap();

        let swapchain_image_views = swapchain_images
            .iter()
            .map(|image| {
                let image_view_create_info = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::B8G8R8A8_UNORM)
                    .components(Default::default())
                    .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1).level_count(1));

                unsafe { device_loader.create_image_view(&image_view_create_info, None) }
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = unsafe { device_loader.create_pipeline_layout(&pipeline_layout_create_info, None) }.unwrap();

        let vertex_shader = create_shader_module(&device_loader, ShaderKind::Vertex, "main", VERTEX_SOURCE);
        let fragment_shader = create_shader_module(&device_loader, ShaderKind::Fragment, "main", FRAGMENT_SOURCE);

        let pipeline = create_pipeline(&device_loader, vertex_shader, "main", fragment_shader, "main", vk::Format::B8G8R8A8_UNORM, pipeline_layout);

        unsafe { device_loader.destroy_shader_module(fragment_shader, None) };
        unsafe { device_loader.destroy_shader_module(vertex_shader, None) };

        let frames: Vec<_> = (0..NUM_FRAMES).into_iter().map(|_| ManuallyDrop::new(Frame::new(device_loader.clone()))).collect();

        Self {
            _entry_loader: entry_loader,

            instance_loader,
            surface_loader,

            surface,

            device_loader,
            swapchain_loader,
            dynamic_rendering_loader,
            synchronization2_loader,

            direct_queue,

            swapchain,
            swapchain_images,
            swapchain_image_views,

            pipeline_layout,
            pipeline,

            frames
        }
    }
}

impl Drop for RenderCtx {
    fn drop(&mut self) {
        unsafe {
            self.device_loader.device_wait_idle().unwrap();

            self.device_loader.destroy_pipeline(self.pipeline, None);
            self.device_loader.destroy_pipeline_layout(self.pipeline_layout, None);

            self.frames.iter_mut().for_each(|frame| ManuallyDrop::drop(frame));

            self.swapchain_image_views.iter().for_each(|image_view| self.device_loader.destroy_image_view(*image_view, None));
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);

            self.device_loader.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance_loader.destroy_instance(None);
        }
    }
}

fn render_frame(ctx: &RenderCtx, frame_index: &mut usize) {
    let device_loader = &ctx.device_loader;
    let swapchain_loader = &ctx.swapchain_loader;

    let direct_queue = ctx.direct_queue;
    let swapchain = ctx.swapchain;

    let current_frame = &ctx.frames[*frame_index];

    let present_semaphore = current_frame.present_semaphore;
    let render_semaphore = current_frame.render_semaphore;

    let fence = current_frame.fence;
    unsafe { device_loader.wait_for_fences(slice::from_ref(&fence), true, u64::MAX) }.unwrap();
    unsafe { device_loader.reset_fences(slice::from_ref(&fence)) }.unwrap();

    let command_pool = current_frame.command_pool;
    let command_buffer = current_frame.command_buffer;

    unsafe { device_loader.reset_command_pool(command_pool, vk::CommandPoolResetFlags::RELEASE_RESOURCES) }.unwrap();

    let image_index = unsafe { swapchain_loader.acquire_next_image(swapchain, u64::MAX, present_semaphore, vk::Fence::null()) }
        .unwrap()
        .0;

    let command_buffer_begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe { device_loader.begin_command_buffer(command_buffer, &command_buffer_begin_info) }.unwrap();

    let image = ctx.swapchain_images[image_index as usize];

    let image_memory_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1));

    unsafe {
        ctx.synchronization2_loader
            .cmd_pipeline_barrier2(command_buffer, &vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&image_memory_barrier)));
    }

    let color_attachment = vk::RenderingAttachmentInfo::default()
        .image_view(ctx.swapchain_image_views[image_index as usize])
        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [100.0 / 255.0, 149.0 / 255.0, 237.0 / 255.0, 1.0]
            }
        });

    let rendering_info = vk::RenderingInfo::default()
        .render_area(vk::Rect2D::default().extent(vk::Extent2D::default().width(1600).height(900)))
        .layer_count(1)
        .color_attachments(slice::from_ref(&color_attachment));

    unsafe { ctx.dynamic_rendering_loader.cmd_begin_rendering(command_buffer, &rendering_info) };

    render_frame_inner(ctx, current_frame);

    unsafe { ctx.dynamic_rendering_loader.cmd_end_rendering(command_buffer) };

    let image_memory_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1));

    unsafe {
        ctx.synchronization2_loader
            .cmd_pipeline_barrier2(command_buffer, &vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&image_memory_barrier)));
    };

    unsafe { device_loader.end_command_buffer(command_buffer) }.unwrap();

    let wait_semaphores = [present_semaphore];
    let wait_dst_stage_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

    let submit_info = vk::SubmitInfo::default()
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_dst_stage_mask)
        .command_buffers(slice::from_ref(&command_buffer))
        .signal_semaphores(slice::from_ref(&render_semaphore));

    unsafe { device_loader.queue_submit(direct_queue, slice::from_ref(&submit_info), fence) }.unwrap();

    let present_info = vk::PresentInfoKHR::default()
        .wait_semaphores(slice::from_ref(&render_semaphore))
        .swapchains(slice::from_ref(&swapchain))
        .image_indices(slice::from_ref(&image_index));

    unsafe { swapchain_loader.queue_present(direct_queue, &present_info) }.unwrap();
}

fn create_shader_module(device: &Device, kind: ShaderKind, entry_point_name: &str, source: &str) -> vk::ShaderModule {
    let compiler = Compiler::new().unwrap();
    let mut compile_options = CompileOptions::new().unwrap();
    compile_options.set_target_spirv(SpirvVersion::V1_4);

    let artifact = compiler.compile_into_spirv(source, kind, "", entry_point_name, Some(&compile_options)).unwrap();

    unsafe {
        let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(artifact.as_binary());

        device.create_shader_module(&shader_module_create_info, None).unwrap()
    }
}

fn create_pipeline(
    device: &Device,
    vertex_shader: vk::ShaderModule,
    vertex_entry_point: &str,
    fragment_shader: vk::ShaderModule,
    fragment_entry_point: &str,
    swapchain_format: vk::Format,
    layout: vk::PipelineLayout
) -> vk::Pipeline {
    let vertex_entry_point = CString::new(vertex_entry_point).unwrap();
    let fragment_entry_point = CString::new(fragment_entry_point).unwrap();

    let shader_stage_create_infos = vec![
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader)
            .name(&vertex_entry_point),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader)
            .name(&fragment_entry_point),
    ];

    let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport = vk::Viewport::default().width(1.0).height(1.0).max_depth(1.0);

    let scissor = vk::Rect2D::default().extent(vk::Extent2D { width: 1, height: 1 });

    let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::default()
        .viewports(slice::from_ref(&viewport))
        .scissors(slice::from_ref(&scissor));

    let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0);

    let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let blend_attachment_state = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA);

    let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::default().attachments(slice::from_ref(&blend_attachment_state));

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let mut pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(slice::from_ref(&swapchain_format));

    let graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stage_create_infos)
        .vertex_input_state(&vertex_input_state_create_info)
        .input_assembly_state(&input_assembly_state_create_info)
        .viewport_state(&viewport_state_create_info)
        .rasterization_state(&rasterization_state_create_info)
        .multisample_state(&multisample_state_create_info)
        .color_blend_state(&color_blend_state_create_info)
        .dynamic_state(&dynamic_state_create_info)
        .layout(layout)
        .push_next(&mut pipeline_rendering_create_info);

    unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), slice::from_ref(&graphics_pipeline_create_info), None) }.unwrap()[0]
}

fn render_frame_inner(ctx: &RenderCtx, current_frame: &Frame) {
    let command_buffer = current_frame.command_buffer;

    unsafe {
        ctx.device_loader.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, ctx.pipeline);

        let viewport = vk::Viewport::default().width(1600.0).height(900.0).max_depth(1.0);
        let scissor = vk::Rect2D::default().extent(vk::Extent2D { width: 1600, height: 900 });

        ctx.device_loader.cmd_set_viewport(command_buffer, 0, slice::from_ref(&viewport));
        ctx.device_loader.cmd_set_scissor(command_buffer, 0, slice::from_ref(&scissor));

        ctx.device_loader.cmd_draw(command_buffer, 3, 1, 0, 0);
    };
}

fn main() {
    let mut event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("ash-triangle")
        .with_inner_size(Size::Physical(PhysicalSize::new(1600, 900)))
        .build(&event_loop)
        .unwrap();

    let render_ctx = RenderCtx::new(&window);

    let mut running = true;
    let mut frame_index = 0;

    while running {
        event_loop.run_return(|event, _, control_flow| {
            *control_flow = ControlFlow::Wait;
            match event {
                Event::WindowEvent { event, window_id } => {
                    if window.id() == window_id {
                        match event {
                            WindowEvent::CloseRequested => running = false,
                            WindowEvent::KeyboardInput { input, .. } => {
                                if let Some(key_code) = input.virtual_keycode {
                                    if key_code == VirtualKeyCode::Escape {
                                        running = false;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Event::MainEventsCleared => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            }
        });

        render_frame(&render_ctx, &mut frame_index);
        frame_index = usize::from(frame_index == 0);
    }
}
