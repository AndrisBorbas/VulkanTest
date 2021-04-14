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
#include "Vulkan/VulkanDevices.hpp"
#include "Vulkan/VulkanInit.hpp"
#include "Vulkan/VulkanRendering.hpp"
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

	vk::PhysicalDevice physicalDevice_;
	vk::Device device_;

	vk::Queue graphicsQueue_;
	vk::Queue presentQueue_;

	vk::SwapchainKHR swapchain_;
	std::vector<vk::Image> swapchainImages_;
	std::vector<vk::ImageView> swapchainImageViews_;
	vk::Format swapchainImageFormat_;
	vk::Extent2D swapchainExtent_;

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
		auto* app                = static_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
		app->framebufferResized_ = true;
	}

	void initVulkan()
	{
		instance_       = createInstance(validationLayers);
		debugMessenger_ = setupDebugMessenger(instance_);

		surface_ = createSurface(instance_, window_);

		physicalDevice_ = pickPhysicalDevice(instance_, surface_, deviceExtensions);
		device_ = createLogicalDevice(instance_, surface_, physicalDevice_, graphicsQueue_, presentQueue_,
									  deviceExtensions, validationLayers);

		swapchain_           = createSwapChain(device_, physicalDevice_, window_, surface_, swapchainImages_,
                                     swapchainImageFormat_, swapchainExtent_, preferredPresentMode);
		swapchainImageViews_ = createImageViews(device_, swapchainImages_, swapchainImageFormat_);

		createRenderPass();
		createGraphicsPipeline();

		createFramebuffers();
		createCommandPool();
		createVertexBuffer();
		createIndexBuffer();
		createCommandBuffers();

		createSyncObjects();
	}

	void createGraphicsPipeline()
	{
		auto vertShaderCode = readFile("assets/shaders/test.vert.spv");
		auto fragShaderCode = readFile("assets/shaders/test.frag.spv");

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
							  .width    = static_cast<float>(swapchainExtent_.width),
							  .height   = static_cast<float>(swapchainExtent_.height),
							  .minDepth = 0.0f,
							  .maxDepth = 1.0f};

		vk::Rect2D scissor{.offset = {0, 0}, .extent = swapchainExtent_};

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
		vk::AttachmentDescription colorAttachment{.format         = swapchainImageFormat_,
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
		swapChainFramebuffers_.resize(swapchainImageViews_.size());
		for (size_t i = 0; i < swapchainImageViews_.size(); i++) {
			vk::ImageView attachments[] = {swapchainImageViews_[i]};

			vk::FramebufferCreateInfo framebufferInfo{.renderPass      = renderPass_,
													  .attachmentCount = 1,
													  .pAttachments    = attachments,
													  .width           = swapchainExtent_.width,
													  .height          = swapchainExtent_.height,
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
			renderPassInfo.renderArea.extent = swapchainExtent_;

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
		imagesInFlight_.resize(swapchainImages_.size(), nullptr);

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

		for (auto& swapChainImageView : swapchainImageViews_) {
			device_.destroyImageView(swapChainImageView, nullptr);
		}

		device_.destroySwapchainKHR(swapchain_, nullptr);
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

		swapchain_           = createSwapChain(device_, physicalDevice_, window_, surface_, swapchainImages_,
                                     swapchainImageFormat_, swapchainExtent_, preferredPresentMode);
		swapchainImageViews_ = createImageViews(device_, swapchainImages_, swapchainImageFormat_);
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
			swapchain_, UINT64_MAX, imageAvailableSemaphores_[currentFrame_], nullptr, &imageIndex);

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

		vk::SwapchainKHR swapChains[] = {swapchain_};

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
