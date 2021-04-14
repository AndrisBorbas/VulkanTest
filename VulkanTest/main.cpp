#include "Defines.h"

#define VK_USE_PLATFORM_WIN32_KHR

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

#include "Shaders.hpp"
#include "VulkanDevices.hpp"
#include "VulkanInit.hpp"
#include "VulkanRendering.hpp"
#include "utils.hpp"

const uint32_t WIDTH  = 800;
const uint32_t HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
const std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

const vk::PresentModeKHR preferredPresentMode = vk::PresentModeKHR::eImmediate;

class HelloTriangleApplication
{
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	GLFWwindow* window_;

	vk::Instance instance_;

	vk::DebugUtilsMessengerEXT debugMessenger_;

	vk::SurfaceKHR surface_;

	vk::PhysicalDevice physicalDevice_ = nullptr;
	vk::Device device_;

	vk::Queue graphicsQueue_;
	vk::Queue presentQueue_;

	vk::SwapchainKHR swapChain_;
	std::vector<vk::Image> swapChainImages_;
	std::vector<vk::ImageView> swapChainImageViews_;
	vk::Format swapChainImageFormat_;
	vk::Extent2D swapChainExtent_;

	vk::RenderPass renderPass_;
	vk::PipelineLayout pipelineLayout_;
	vk::Pipeline graphicsPipeline_;

	std::vector<vk::Framebuffer> swapChainFramebuffers_;

	vk::CommandPool commandPool_;
	std::vector<vk::CommandBuffer> commandBuffers_;

	std::vector<vk::Semaphore> imageAvailableSemaphores_;
	std::vector<vk::Semaphore> renderFinishedSemaphores_;
	std::vector<vk::Fence> inFlightFences_;
	std::vector<vk::Fence> imagesInFlight_;

	const std::vector<Vertex> vertices_  = {{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
                                           {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
                                           {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
                                           {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}};
	const std::vector<uint16_t> indices_ = {0, 1, 2, 2, 3, 0};

	vk::Buffer vertexBuffer_;
	vk::DeviceMemory vertexBufferMemory_;
	vk::Buffer indexBuffer_;
	vk::DeviceMemory indexBufferMemory_;

	size_t currentFrame_ = 0;

	bool framebufferResized_ = false;

	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window_ = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window_, this);
		glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
	{
		auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
		app->framebufferResized_ = true;
	}

	void initVulkan()
	{
		instance_       = createInstance(validationLayers);
		debugMessenger_ = setupDebugMessenger(instance_);

		createSurface();

		pickPhysicalDevice();
		createLogicalDevice();

		createSwapChain();
		createImageViews();

		createRenderPass();
		createGraphicsPipeline();

		createFramebuffers();
		createCommandPool();
		createVertexBuffer();
		createIndexBuffer();
		createCommandBuffers();

		createSyncObjects();
	}

	void createSurface()
	{
		VkSurfaceKHR tempSurface;
		if (glfwCreateWindowSurface(instance_, window_, nullptr, &tempSurface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
		surface_ = vk::SurfaceKHR(tempSurface);
	}

	void pickPhysicalDevice()
	{
		uint32_t deviceCount = 0;
		instance_.enumeratePhysicalDevices(&deviceCount, nullptr);
		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}
		std::vector<vk::PhysicalDevice> devices(deviceCount);
		instance_.enumeratePhysicalDevices(&deviceCount, devices.data());

		// Use an ordered map to automatically sort candidates by increasing score
		std::multimap<int, VkPhysicalDevice> candidates;

		for (const auto& item : devices) {
			int score = rateDeviceSuitability(item, surface_, deviceExtensions);
			candidates.insert(std::make_pair(score, item));
		}

		// Check if the best candidate is suitable at all
		if (candidates.rbegin()->first > 0) {
			physicalDevice_ = candidates.rbegin()->second;
			std::cout << physicalDevice_.getProperties().deviceName << std::endl;
		} else {
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	void createLogicalDevice()
	{
		QueueFamilyIndices queueFamilies = findQueueFamilies(physicalDevice_, surface_);

		std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = {queueFamilies.graphicsFamily.value(),
												  queueFamilies.presentFamily.value()};

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			vk::DeviceQueueCreateInfo queueCreateInfo{
				.queueFamilyIndex = queueFamily, .queueCount = 1, .pQueuePriorities = &queuePriority};
			queueCreateInfos.push_back(queueCreateInfo);
		}

		vk::PhysicalDeviceFeatures deviceFeatures{};

		vk::DeviceCreateInfo createInfo{
			.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size()),
			.pQueueCreateInfos       = queueCreateInfos.data(),
			.enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size()),
			.ppEnabledExtensionNames = deviceExtensions.data(),
			.pEnabledFeatures        = &deviceFeatures};

		if (enableValidationLayers) {
			createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		} else {
			createInfo.enabledLayerCount = 0;
		}

		if (physicalDevice_.createDevice(&createInfo, nullptr, &device_) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create logical device!");
		}

		device_.getQueue(queueFamilies.graphicsFamily.value(), 0, &graphicsQueue_);
		device_.getQueue(queueFamilies.presentFamily.value(), 0, &presentQueue_);
	}

	void createSwapChain()
	{
		const SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice_, surface_);

		const vk::PresentModeKHR presentMode =
			chooseSwapPresentMode(swapChainSupport.presentModes, preferredPresentMode);
		const vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		const vk::Extent2D extent                = chooseSwapExtent(swapChainSupport.capabilities, window_);

		std::cout << std::endl << to_string(presentMode) << std::endl;

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

		if (swapChainSupport.capabilities.maxImageCount > 0
			&& imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		vk::SwapchainCreateInfoKHR createInfo{.surface          = surface_,
											  .minImageCount    = imageCount,
											  .imageFormat      = surfaceFormat.format,
											  .imageColorSpace  = surfaceFormat.colorSpace,
											  .imageExtent      = extent,
											  .imageArrayLayers = 1,
											  .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment};

		QueueFamilyIndices queueFamilies = findQueueFamilies(physicalDevice_, surface_);
		uint32_t queueFamilyIndices[]    = {queueFamilies.graphicsFamily.value(),
                                         queueFamilies.presentFamily.value()};

		if (queueFamilies.graphicsFamily != queueFamilies.presentFamily) {
			createInfo.imageSharingMode      = vk::SharingMode::eConcurrent;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices   = queueFamilyIndices;
		} else {
			createInfo.imageSharingMode      = vk::SharingMode::eExclusive;
			createInfo.queueFamilyIndexCount = 0;        // Optional
			createInfo.pQueueFamilyIndices   = nullptr;  // Optional
		}

		createInfo.preTransform   = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;

		createInfo.presentMode = presentMode;
		createInfo.clipped     = VK_TRUE;

		createInfo.oldSwapchain = nullptr;

		if (device_.createSwapchainKHR(&createInfo, nullptr, &swapChain_) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create swap chain!");
		}

		swapChainImageFormat_ = surfaceFormat.format;
		swapChainExtent_      = extent;

		device_.getSwapchainImagesKHR(swapChain_, &imageCount, nullptr);
		swapChainImages_.resize(imageCount);
		device_.getSwapchainImagesKHR(swapChain_, &imageCount, swapChainImages_.data());
	}

	void createImageViews()
	{
		swapChainImageViews_.resize(swapChainImages_.size());
		for (size_t i = 0; i < swapChainImages_.size(); i++) {
			vk::ImageViewCreateInfo createInfo{
				.image    = swapChainImages_[i],
				.viewType = vk::ImageViewType::e2D,
				.format   = swapChainImageFormat_,
			};

			createInfo.components.r = vk::ComponentSwizzle::eIdentity;
			createInfo.components.g = vk::ComponentSwizzle::eIdentity;
			createInfo.components.b = vk::ComponentSwizzle::eIdentity;
			createInfo.components.a = vk::ComponentSwizzle::eIdentity;

			createInfo.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
			createInfo.subresourceRange.baseMipLevel   = 0;
			createInfo.subresourceRange.levelCount     = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount     = 1;

			if (device_.createImageView(&createInfo, nullptr, &swapChainImageViews_[i])
				!= vk::Result::eSuccess) {
				throw std::runtime_error("failed to create image views!");
			}
		}
	}

	void createGraphicsPipeline()
	{
		auto vertShaderCode = readFile("assets/shaders/shader.vert.spv");
		auto fragShaderCode = readFile("assets/shaders/shader.frag.spv");

		vk::ShaderModule vertShaderModule = createShaderModule(device_, vertShaderCode);
		vk::ShaderModule fragShaderModule = createShaderModule(device_, fragShaderCode);

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex,
															  .pName = "main"};
		vertShaderStageInfo.module = vertShaderModule;

		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment,
															  .pName = "main"};
		fragShaderStageInfo.module = fragShaderModule;

		vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		auto bindingDescription    = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
			.vertexBindingDescriptionCount   = 1,
			.pVertexBindingDescriptions      = &bindingDescription,
			.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
			.pVertexAttributeDescriptions    = attributeDescriptions.data()};

		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
			.topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = VK_FALSE};

		vk::Viewport viewport{.x        = 0.0f,
							  .y        = 0.0f,
							  .width    = static_cast<float>(swapChainExtent_.width),
							  .height   = static_cast<float>(swapChainExtent_.height),
							  .minDepth = 0.0f,
							  .maxDepth = 1.0f};

		vk::Rect2D scissor{.offset = {0, 0}, .extent = swapChainExtent_};

		vk::PipelineViewportStateCreateInfo viewportState{
			.viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor};

		vk::PipelineRasterizationStateCreateInfo rasterizer{.depthClampEnable        = VK_FALSE,
															.rasterizerDiscardEnable = VK_FALSE,
															.polygonMode             = vk::PolygonMode::eFill,
															.cullMode        = vk::CullModeFlagBits::eBack,
															.frontFace       = vk::FrontFace::eClockwise,
															.depthBiasEnable = VK_FALSE,
															.depthBiasConstantFactor = 0.0f,
															.depthBiasClamp          = 0.0f,
															.depthBiasSlopeFactor    = 0.0f,
															.lineWidth               = 1.0f};

		vk::PipelineMultisampleStateCreateInfo multisampling{
			.rasterizationSamples  = vk::SampleCountFlagBits::e1,
			.sampleShadingEnable   = true,
			.minSampleShading      = 1.0f,
			.pSampleMask           = nullptr,
			.alphaToCoverageEnable = false,
			.alphaToOneEnable      = false};

		vk::PipelineColorBlendAttachmentState colorBlendAttachment{
			.blendEnable         = VK_TRUE,
			.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
			.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
			.colorBlendOp        = vk::BlendOp::eAdd,
			.srcAlphaBlendFactor = vk::BlendFactor::eOne,
			.dstAlphaBlendFactor = vk::BlendFactor::eZero,
			.alphaBlendOp        = vk::BlendOp::eAdd,
			.colorWriteMask      = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
							  | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

		vk::PipelineColorBlendStateCreateInfo colorBlending{.logicOpEnable   = VK_FALSE,
															.logicOp         = vk::LogicOp::eCopy,
															.attachmentCount = 1,
															.pAttachments    = &colorBlendAttachment};
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		vk::DynamicState dynamicStates[] = {vk::DynamicState::eViewport, vk::DynamicState::eLineWidth};

		vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = 2,
														.pDynamicStates    = dynamicStates};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount         = 0,
														.pSetLayouts            = nullptr,
														.pushConstantRangeCount = 0,
														.pPushConstantRanges    = nullptr};

		if (device_.createPipelineLayout(&pipelineLayoutInfo, nullptr, &pipelineLayout_)
			!= vk::Result::eSuccess) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		vk::GraphicsPipelineCreateInfo pipelineInfo{.stageCount          = 2,
													.pStages             = shaderStages,
													.pVertexInputState   = &vertexInputInfo,
													.pInputAssemblyState = &inputAssembly,
													.pViewportState      = &viewportState,
													.pRasterizationState = &rasterizer,
													.pMultisampleState   = &multisampling,
													.pDepthStencilState  = nullptr,
													.pColorBlendState    = &colorBlending,
													.pDynamicState       = nullptr,
													.layout              = pipelineLayout_,
													.renderPass          = renderPass_,
													.subpass             = 0,
													.basePipelineHandle  = nullptr,
													.basePipelineIndex   = -1};

		if (device_.createGraphicsPipelines(nullptr, 1, &pipelineInfo, nullptr, &graphicsPipeline_)
			!= vk::Result::eSuccess) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		device_.destroyShaderModule(fragShaderModule, nullptr);
		device_.destroyShaderModule(vertShaderModule, nullptr);
	}

	void createRenderPass()
	{
		vk::AttachmentDescription colorAttachment{.format         = swapChainImageFormat_,
												  .samples        = vk::SampleCountFlagBits::e1,
												  .loadOp         = vk::AttachmentLoadOp::eClear,
												  .storeOp        = vk::AttachmentStoreOp::eStore,
												  .stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
												  .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
												  .initialLayout  = vk::ImageLayout::eUndefined,
												  .finalLayout    = vk::ImageLayout::ePresentSrcKHR};

		vk::AttachmentReference colorAttachmentRef{.attachment = 0,
												   .layout     = vk::ImageLayout::eColorAttachmentOptimal};

		vk::SubpassDescription subpass{.pipelineBindPoint    = vk::PipelineBindPoint::eGraphics,
									   .colorAttachmentCount = 1,
									   .pColorAttachments    = &colorAttachmentRef};

		vk::SubpassDependency dependency{.srcSubpass    = VK_SUBPASS_EXTERNAL,
										 .dstSubpass    = 0,
										 .srcStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput,
										 .dstStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput,
										 .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite};

		vk::RenderPassCreateInfo renderPassInfo{.attachmentCount = 1,
												.pAttachments    = &colorAttachment,
												.subpassCount    = 1,
												.pSubpasses      = &subpass,
												.dependencyCount = 1,
												.pDependencies   = &dependency};

		if (device_.createRenderPass(&renderPassInfo, nullptr, &renderPass_) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create render pass!");
		}
	}

	void createFramebuffers()
	{
		swapChainFramebuffers_.resize(swapChainImageViews_.size());
		for (size_t i = 0; i < swapChainImageViews_.size(); i++) {
			vk::ImageView attachments[] = {swapChainImageViews_[i]};

			vk::FramebufferCreateInfo framebufferInfo{.renderPass      = renderPass_,
													  .attachmentCount = 1,
													  .pAttachments    = attachments,
													  .width           = swapChainExtent_.width,
													  .height          = swapChainExtent_.height,
													  .layers          = 1};
			if (device_.createFramebuffer(&framebufferInfo, nullptr, &swapChainFramebuffers_[i])
				!= vk::Result::eSuccess) {
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}

	void createCommandPool()
	{
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice_, surface_);

		vk::CommandPoolCreateInfo poolInfo{.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()};

		if (device_.createCommandPool(&poolInfo, nullptr, &commandPool_) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create command pool!");
		}
	}

	void createBuffer(vk::DeviceSize size,
					  vk::BufferUsageFlags usage,
					  vk::MemoryPropertyFlags properties,
					  vk::Buffer& buffer,
					  vk::DeviceMemory& bufferMemory)
	{
		vk::BufferCreateInfo bufferInfo = {
			.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};

		if (device_.createBuffer(&bufferInfo, nullptr, &buffer) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create buffer!");
		}

		vk::MemoryRequirements memRequirements;
		device_.getBufferMemoryRequirements(buffer, &memRequirements);

		vk::MemoryAllocateInfo allocInfo{
			.allocationSize  = memRequirements.size,
			.memoryTypeIndex = findMemoryType(physicalDevice_, memRequirements.memoryTypeBits, properties)};

		if (device_.allocateMemory(&allocInfo, nullptr, &bufferMemory) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to allocate buffer memory!");
		}

		vkBindBufferMemory(device_, buffer, bufferMemory, 0);
	}

	void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
	{
		vk::CommandBufferAllocateInfo allocInfo{
			.commandPool = commandPool_, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};

		vk::CommandBuffer commandBuffer;
		device_.allocateCommandBuffers(&allocInfo, &commandBuffer);

		vk::CommandBufferBeginInfo beginInfo = {.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};

		commandBuffer.begin(&beginInfo);

		vk::BufferCopy copyRegion{.srcOffset = 0,  // Optional
								  .dstOffset = 0,  // Optional
								  .size      = size};

		commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

		commandBuffer.end();

		vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &commandBuffer};

		graphicsQueue_.submit(1, &submitInfo, {});
		graphicsQueue_.waitIdle();

		device_.freeCommandBuffers(commandPool_, 1, &commandBuffer);
	}

	void createVertexBuffer()
	{
		const vk::DeviceSize bufferSize = sizeof(vertices_[0]) * vertices_.size();

		vk::Buffer stagingBuffer;
		vk::DeviceMemory stagingBufferMemory;
		createBuffer(bufferSize,
					 vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eVertexBuffer,
					 vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

		void* data;
		device_.mapMemory(stagingBufferMemory, 0, bufferSize, {}, &data);
		memcpy(data, vertices_.data(), static_cast<size_t>(bufferSize));
		device_.unmapMemory(stagingBufferMemory);

		createBuffer(bufferSize,
					 vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
					 vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer_, vertexBufferMemory_);

		copyBuffer(stagingBuffer, vertexBuffer_, bufferSize);

		device_.destroyBuffer(stagingBuffer, nullptr);
		device_.freeMemory(stagingBufferMemory, nullptr);
	}

	void createIndexBuffer()
	{
		const vk::DeviceSize bufferSize = sizeof(indices_[0]) * indices_.size();

		vk::Buffer stagingBuffer;
		vk::DeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
					 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
					 stagingBuffer, stagingBufferMemory);

		void* data;
		device_.mapMemory(stagingBufferMemory, 0, bufferSize, {}, &data);
		memcpy(data, indices_.data(), static_cast<size_t>(bufferSize));
		device_.unmapMemory(stagingBufferMemory);

		createBuffer(bufferSize,
					 vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
					 vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer_, indexBufferMemory_);

		copyBuffer(stagingBuffer, indexBuffer_, bufferSize);

		device_.destroyBuffer(stagingBuffer, nullptr);
		device_.freeMemory(stagingBufferMemory, nullptr);
	}

	void createCommandBuffers()
	{
		commandBuffers_.resize(swapChainFramebuffers_.size());
		vk::CommandBufferAllocateInfo allocInfo{
			.commandPool        = commandPool_,
			.level              = vk::CommandBufferLevel::ePrimary,
			.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size())};

		if (device_.allocateCommandBuffers(&allocInfo, commandBuffers_.data()) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to allocate command buffers!");
		}

		for (size_t i = 0; i < commandBuffers_.size(); i++) {
			vk::CommandBufferBeginInfo beginInfo{};
			if (commandBuffers_[i].begin(&beginInfo) != vk::Result::eSuccess) {
				throw std::runtime_error("failed to begin recording command buffer!");
			}

			vk::ClearValue clearColor = {std::array<float, 4>({{0.0f, 0.0f, 0.0f, 1.0f}})};

			vk::RenderPassBeginInfo renderPassInfo{.renderPass      = renderPass_,
												   .framebuffer     = swapChainFramebuffers_[i],
												   .clearValueCount = 1,
												   .pClearValues    = &clearColor};

			renderPassInfo.renderArea.offset = VkOffset2D{0, 0};
			renderPassInfo.renderArea.extent = swapChainExtent_;

			commandBuffers_[i].beginRenderPass(&renderPassInfo, vk::SubpassContents::eInline);

			commandBuffers_[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline_);

			vk::Buffer vertexBuffers[] = {vertexBuffer_};
			vk::DeviceSize offsets[]   = {0};
			commandBuffers_[i].bindVertexBuffers(0, 1, vertexBuffers, offsets);
			commandBuffers_[i].bindIndexBuffer(indexBuffer_, 0, vk::IndexType::eUint16);

			commandBuffers_[i].drawIndexed(static_cast<uint32_t>(indices_.size()), 1, 0, 0, 0);

			commandBuffers_[i].endRenderPass();

			// Presumably has built in fail check
			commandBuffers_[i].end();
		}
	}

	void createSyncObjects()
	{
		imageAvailableSemaphores_.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores_.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences_.resize(MAX_FRAMES_IN_FLIGHT);
		imagesInFlight_.resize(swapChainImages_.size(), nullptr);

		vk::SemaphoreCreateInfo semaphoreInfo{};
		vk::FenceCreateInfo fenceInfo{.flags = vk::FenceCreateFlagBits::eSignaled};

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (device_.createSemaphore(&semaphoreInfo, nullptr, &imageAvailableSemaphores_[i])
					!= vk::Result::eSuccess
				|| device_.createSemaphore(&semaphoreInfo, nullptr, &renderFinishedSemaphores_[i])
					   != vk::Result::eSuccess
				|| device_.createFence(&fenceInfo, nullptr, &inFlightFences_[i]) != vk::Result::eSuccess) {
				throw std::runtime_error("failed to create semaphores for a frame!");
			}
		}
	}

	void cleanupSwapChain()
	{
		for (auto& swapChainFramebuffer : swapChainFramebuffers_) {
			device_.destroyFramebuffer(swapChainFramebuffer, nullptr);
		}

		device_.freeCommandBuffers(commandPool_, static_cast<uint32_t>(commandBuffers_.size()),
								   commandBuffers_.data());

		device_.destroyPipeline(graphicsPipeline_, nullptr);
		device_.destroyPipelineLayout(pipelineLayout_, nullptr);
		device_.destroyRenderPass(renderPass_, nullptr);

		for (auto& swapChainImageView : swapChainImageViews_) {
			device_.destroyImageView(swapChainImageView, nullptr);
		}

		device_.destroySwapchainKHR(swapChain_, nullptr);
	}

	void recreateSwapChain()
	{
		int width = 0, height = 0;
		glfwGetFramebufferSize(window_, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window_, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(device_);

		cleanupSwapChain();

		createSwapChain();
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createFramebuffers();
		createCommandBuffers();
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window_)) {
			glfwPollEvents();
			drawFrame();
		}

		device_.waitIdle();
	}

	void drawFrame()
	{
		device_.waitForFences(1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		vk::Result result = device_.acquireNextImageKHR(
			swapChain_, UINT64_MAX, imageAvailableSemaphores_[currentFrame_], nullptr, &imageIndex);

		if (result == vk::Result::eErrorOutOfDateKHR) {
			recreateSwapChain();
			return;
		} else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		// Check if a previous frame is using this image (i.e. there is its fence to wait on)
		if (imagesInFlight_[imageIndex] != VK_NULL_HANDLE) {
			device_.waitForFences(1, &imagesInFlight_[imageIndex], VK_TRUE, UINT64_MAX);
		}
		// Mark the image as now being in use by this frame
		imagesInFlight_[imageIndex] = inFlightFences_[currentFrame_];

		vk::Semaphore waitSemaphores[]      = {imageAvailableSemaphores_[currentFrame_]};
		vk::Semaphore signalSemaphores[]    = {renderFinishedSemaphores_[currentFrame_]};
		vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};

		vk::SubmitInfo submitInfo{.waitSemaphoreCount   = 1,
								  .pWaitSemaphores      = waitSemaphores,
								  .pWaitDstStageMask    = waitStages,
								  .commandBufferCount   = 1,
								  .pCommandBuffers      = &commandBuffers_[imageIndex],
								  .signalSemaphoreCount = 1,
								  .pSignalSemaphores    = signalSemaphores};

		device_.resetFences(1, &inFlightFences_[currentFrame_]);

		if (graphicsQueue_.submit(1, &submitInfo, inFlightFences_[currentFrame_]) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		vk::SwapchainKHR swapChains[] = {swapChain_};

		vk::PresentInfoKHR presentInfo{.waitSemaphoreCount = 1,
									   .pWaitSemaphores    = signalSemaphores,
									   .swapchainCount     = 1,
									   .pSwapchains        = swapChains,
									   .pImageIndices      = &imageIndex,
									   .pResults           = nullptr};

		result = presentQueue_.presentKHR(&presentInfo);

		if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR
			|| framebufferResized_) {
			framebufferResized_ = false;
			recreateSwapChain();
		} else if (result != vk::Result::eSuccess) {
			throw std::runtime_error("failed to present swap chain image!");
		}

		currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void cleanup()
	{
		std::cout << std::endl;
		cleanupSwapChain();

		device_.destroyBuffer(indexBuffer_, nullptr);
		device_.freeMemory(indexBufferMemory_, nullptr);

		device_.destroyBuffer(vertexBuffer_, nullptr);
		device_.freeMemory(vertexBufferMemory_, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			device_.destroySemaphore(renderFinishedSemaphores_[i], nullptr);
			device_.destroySemaphore(imageAvailableSemaphores_[i], nullptr);
			device_.destroyFence(inFlightFences_[i], nullptr);
		}

		device_.destroyCommandPool(commandPool_, nullptr);

		device_.destroy();

		instance_.destroySurfaceKHR(surface_, nullptr);

#ifdef ENABLE_VALIDATION_LAYERS
		DestroyDebugUtilsMessengerEXT(instance_, debugMessenger_, nullptr);
#endif

		instance_.destroy();

		glfwDestroyWindow(window_);

		glfwTerminate();
	}
};

int main()
{
	HelloTriangleApplication app;

	try {
		app.run();
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
