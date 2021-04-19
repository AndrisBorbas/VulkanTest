#include "../Defines.h"

#include "VulkanRendering.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>

#include "../utils.hpp"
#include "VulkanDevices.hpp"

vk::ShaderModule createShaderModule(vk::Device device, const std::vector<char>& code)
{
	vk::ShaderModuleCreateInfo createInfo;
	createInfo.codeSize = code.size();
	createInfo.pCode    = reinterpret_cast<const uint32_t*>(code.data());
	vk::ShaderModule shaderModule;
	if (device.createShaderModule(&createInfo, nullptr, &shaderModule) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create shader module!");
	}
	return shaderModule;
}

void createBuffer(vk::Device& device,
				  vk::PhysicalDevice& physicalDevice,
				  vk::DeviceSize size,
				  vk::BufferUsageFlags usage,
				  vk::MemoryPropertyFlags properties,
				  vk::Buffer& buffer,
				  vk::DeviceMemory& bufferMemory)
{
	vk::BufferCreateInfo bufferInfo = {
		.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};

	if (device.createBuffer(&bufferInfo, nullptr, &buffer) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create buffer!");
	}

	vk::MemoryRequirements memRequirements;
	device.getBufferMemoryRequirements(buffer, &memRequirements);

	vk::MemoryAllocateInfo allocInfo{
		.allocationSize  = memRequirements.size,
		.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties)};

	if (device.allocateMemory(&allocInfo, nullptr, &bufferMemory) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to allocate buffer memory!");
	}

	device.bindBufferMemory(buffer, bufferMemory, 0);
}

void copyBuffer(vk::Device& device,
				vk::CommandPool& commandPool,
				vk::Queue& graphicsQueue,
				vk::Buffer& srcBuffer,
				vk::Buffer& dstBuffer,
				vk::DeviceSize size)
{
	vk::CommandBufferAllocateInfo allocInfo{
		.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};

	vk::CommandBuffer commandBuffer;
	device.allocateCommandBuffers(&allocInfo, &commandBuffer);

	vk::CommandBufferBeginInfo beginInfo = {.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};

	commandBuffer.begin(&beginInfo);

	vk::BufferCopy copyRegion{.srcOffset = 0,  // Optional
							  .dstOffset = 0,  // Optional
							  .size      = size};

	commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

	commandBuffer.end();

	vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &commandBuffer};

	graphicsQueue.submit(1, &submitInfo, {});
	graphicsQueue.waitIdle();

	device.freeCommandBuffers(commandPool, 1, &commandBuffer);
}

vk::RenderPass createRenderPass(vk::Device& device, vk::Format& swapchainImageFormat)
{
	vk::AttachmentDescription colorAttachment{.format         = swapchainImageFormat,
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

	vk::RenderPass renderPass;

	if (device.createRenderPass(&renderPassInfo, nullptr, &renderPass) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create render pass!");
	}

	return renderPass;
}

vk::DescriptorSetLayout createDescriptorSetLayout(vk::Device& device)
{
	vk::DescriptorSetLayoutBinding uboLayoutBinding{.binding            = 0,
													.descriptorType     = vk::DescriptorType::eUniformBuffer,
													.descriptorCount    = 1,
													.stageFlags         = vk::ShaderStageFlagBits::eVertex,
													.pImmutableSamplers = nullptr};

	vk::DescriptorSetLayout descriptorSetLayout;

	vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = 1, .pBindings = &uboLayoutBinding};

	if (device.createDescriptorSetLayout(&layoutInfo, nullptr, &descriptorSetLayout)
		!= vk::Result::eSuccess) {
		throw std::runtime_error("failed to create descriptor set layout!");
	}

	return descriptorSetLayout;
}

vk::Pipeline createGraphicsPipeline(vk::Device& device,
									vk::Extent2D& swapchainExtent,
									vk::DescriptorSetLayout& descriptorSetLayout,
									vk::PipelineLayout& pipelineLayout,
									vk::RenderPass& renderPass)
{
	auto vertShaderCode = readFile("assets/shaders/test.vert.spv");
	auto fragShaderCode = readFile("assets/shaders/test.frag.spv");

	vk::ShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
	vk::ShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);

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

	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList,
														   .primitiveRestartEnable = VK_FALSE};

	vk::Viewport viewport{.x        = 0.0f,
						  .y        = 0.0f,
						  .width    = static_cast<float>(swapchainExtent.width),
						  .height   = static_cast<float>(swapchainExtent.height),
						  .minDepth = 0.0f,
						  .maxDepth = 1.0f};

	vk::Rect2D scissor{.offset = {0, 0}, .extent = swapchainExtent};

	vk::PipelineViewportStateCreateInfo viewportState{
		.viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor};

	vk::PipelineRasterizationStateCreateInfo rasterizer{.depthClampEnable        = VK_FALSE,
														.rasterizerDiscardEnable = VK_FALSE,
														.polygonMode             = vk::PolygonMode::eFill,
														.cullMode        = vk::CullModeFlagBits::eBack,
														.frontFace       = vk::FrontFace::eCounterClockwise,
														.depthBiasEnable = VK_FALSE,
														.depthBiasConstantFactor = 0.0f,
														.depthBiasClamp          = 0.0f,
														.depthBiasSlopeFactor    = 0.0f,
														.lineWidth               = 1.0f};

	vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples  = vk::SampleCountFlagBits::e1,
														 .sampleShadingEnable   = VK_TRUE,
														 .minSampleShading      = 1.0f,
														 .pSampleMask           = nullptr,
														 .alphaToCoverageEnable = VK_FALSE,
														 .alphaToOneEnable      = VK_FALSE};

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

	vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = 2, .pDynamicStates = dynamicStates};

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount         = 1,
													.pSetLayouts            = &descriptorSetLayout,
													.pushConstantRangeCount = 0,
													.pPushConstantRanges    = nullptr};

	if (device.createPipelineLayout(&pipelineLayoutInfo, nullptr, &pipelineLayout) != vk::Result::eSuccess) {
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
												.layout              = pipelineLayout,
												.renderPass          = renderPass,
												.subpass             = 0,
												.basePipelineHandle  = nullptr,
												.basePipelineIndex   = -1};

	vk::Pipeline graphicsPipeline;

	if (device.createGraphicsPipelines(nullptr, 1, &pipelineInfo, nullptr, &graphicsPipeline)
		!= vk::Result::eSuccess) {
		throw std::runtime_error("failed to create graphics pipeline!");
	}

	device.destroyShaderModule(fragShaderModule, nullptr);
	device.destroyShaderModule(vertShaderModule, nullptr);

	return graphicsPipeline;
}

void createVertexBuffer(vk::Device& device,
						vk::PhysicalDevice& physicalDevice,
						vk::CommandPool& commandPool,
						vk::Queue& graphicsQueue,
						const std::vector<Vertex>& vertices,
						vk::Buffer& vertexBuffer,
						vk::DeviceMemory& vertexBufferMemory)
{
	const vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

	vk::Buffer stagingBuffer;
	vk::DeviceMemory stagingBufferMemory;
	createBuffer(device, physicalDevice, bufferSize,
				 vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eVertexBuffer,
				 vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

	void* data;
	device.mapMemory(stagingBufferMemory, 0, bufferSize, {}, &data);
	memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
	device.unmapMemory(stagingBufferMemory);

	createBuffer(device, physicalDevice, bufferSize,
				 vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
				 vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

	copyBuffer(device, commandPool, graphicsQueue, stagingBuffer, vertexBuffer, bufferSize);

	device.destroyBuffer(stagingBuffer, nullptr);
	device.freeMemory(stagingBufferMemory, nullptr);
}

void createIndexBuffer(vk::Device& device,
					   vk::PhysicalDevice& physicalDevice,
					   vk::CommandPool& commandPool,
					   vk::Queue& graphicsQueue,
					   const std::vector<uint16_t>& indices,
					   vk::Buffer& indexBuffer,
					   vk::DeviceMemory& indexBufferMemory)
{
	const vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

	vk::Buffer stagingBuffer;
	vk::DeviceMemory stagingBufferMemory;
	createBuffer(device, physicalDevice, bufferSize,
				 vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eIndexBuffer,
				 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
				 stagingBuffer, stagingBufferMemory);

	void* data;
	device.mapMemory(stagingBufferMemory, 0, bufferSize, {}, &data);
	memcpy(data, indices.data(), static_cast<size_t>(bufferSize));
	device.unmapMemory(stagingBufferMemory);

	createBuffer(device, physicalDevice, bufferSize,
				 vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
				 vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

	copyBuffer(device, commandPool, graphicsQueue, stagingBuffer, indexBuffer, bufferSize);

	device.destroyBuffer(stagingBuffer, nullptr);
	device.freeMemory(stagingBufferMemory, nullptr);
}

void createUniformBuffers(vk::Device& device,
						  vk::PhysicalDevice& physicalDevice,
						  std::vector<vk::Buffer>& uniformBuffers,
						  std::vector<vk::DeviceMemory>& uniformBuffersMemory,
						  std::vector<vk::Image>& swapchainImages)
{
	vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

	uniformBuffers.resize(swapchainImages.size());
	uniformBuffersMemory.resize(swapchainImages.size());

	for (size_t i = 0; i < swapchainImages.size(); i++) {
		createBuffer(device, physicalDevice, bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
					 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
					 uniformBuffers[i], uniformBuffersMemory[i]);
	}
}

void createFramebuffers(vk::Device& device,
						std::vector<vk::Framebuffer>& swapChainFramebuffers,
						std::vector<vk::ImageView>& swapchainImageViews,
						vk::RenderPass& renderPass,
						vk::Extent2D& swapchainExtent)
{
	swapChainFramebuffers.resize(swapchainImageViews.size());
	for (size_t i = 0; i < swapchainImageViews.size(); i++) {
		vk::ImageView attachments[] = {swapchainImageViews[i]};

		vk::FramebufferCreateInfo framebufferInfo{.renderPass      = renderPass,
												  .attachmentCount = 1,
												  .pAttachments    = attachments,
												  .width           = swapchainExtent.width,
												  .height          = swapchainExtent.height,
												  .layers          = 1};
		if (device.createFramebuffer(&framebufferInfo, nullptr, &swapChainFramebuffers[i])
			!= vk::Result::eSuccess) {
			throw std::runtime_error("failed to create framebuffer!");
		}
	}
}

vk::CommandPool createCommandPool(vk::Device& device,
								  vk::PhysicalDevice& physicalDevice,
								  vk::SurfaceKHR& surface)
{
	QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice, surface);

	vk::CommandPoolCreateInfo poolInfo{.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()};

	vk::CommandPool commandPool;

	if (device.createCommandPool(&poolInfo, nullptr, &commandPool) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create command pool!");
	}
	return commandPool;
}

void createCommandBuffers(vk::Device& device,
						  std::vector<vk::CommandBuffer>& commandBuffers,
						  std::vector<vk::Framebuffer>& swapChainFramebuffers,
						  vk::CommandPool& commandPool,
						  vk::RenderPass& renderPass,
						  vk::Extent2D& swapchainExtent,
						  vk::Pipeline& graphicsPipeline,
						  vk::Buffer& vertexBuffer,
						  vk::Buffer& indexBuffer,
						  vk::PipelineLayout& pipelineLayout,
						  std::vector<vk::DescriptorSet>& descriptorSets,
						  const std::vector<uint16_t>& indices)
{
	commandBuffers.resize(swapChainFramebuffers.size());
	vk::CommandBufferAllocateInfo allocInfo{
		.commandPool        = commandPool,
		.level              = vk::CommandBufferLevel::ePrimary,
		.commandBufferCount = static_cast<uint32_t>(commandBuffers.size())};

	if (device.allocateCommandBuffers(&allocInfo, commandBuffers.data()) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to allocate command buffers!");
	}

	for (size_t i = 0; i < commandBuffers.size(); i++) {
		vk::CommandBufferBeginInfo beginInfo{};
		if (commandBuffers[i].begin(&beginInfo) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		vk::ClearValue clearColor = {std::array<float, 4>({{0.0f, 0.0f, 0.0f, 1.0f}})};

		vk::RenderPassBeginInfo renderPassInfo{.renderPass      = renderPass,
											   .framebuffer     = swapChainFramebuffers[i],
											   .clearValueCount = 1,
											   .pClearValues    = &clearColor};

		renderPassInfo.renderArea.offset = VkOffset2D{0, 0};
		renderPassInfo.renderArea.extent = swapchainExtent;

		commandBuffers[i].beginRenderPass(&renderPassInfo, vk::SubpassContents::eInline);

		commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

		vk::Buffer vertexBuffers[] = {vertexBuffer};
		vk::DeviceSize offsets[]   = {0};
		commandBuffers[i].bindVertexBuffers(0, 1, vertexBuffers, offsets);
		commandBuffers[i].bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint16);

		commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1,
											 &descriptorSets[i], 0, nullptr);
		commandBuffers[i].drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

		commandBuffers[i].endRenderPass();

		// Presumably has built in fail check
		commandBuffers[i].end();
	}
}

void createSyncObjects(vk::Device& device,
					   std::vector<vk::Semaphore>& imageAvailableSemaphores,
					   std::vector<vk::Semaphore>& renderFinishedSemaphores,
					   std::vector<vk::Fence>& inFlightFences,
					   std::vector<vk::Fence>& imagesInFlight,
					   const int MAX_FRAMES_IN_FLIGHT,
					   std::vector<vk::Image>& swapchainImages)
{
	imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
	imagesInFlight.resize(swapchainImages.size(), nullptr);

	vk::SemaphoreCreateInfo semaphoreInfo{};
	vk::FenceCreateInfo fenceInfo{.flags = vk::FenceCreateFlagBits::eSignaled};

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		if (device.createSemaphore(&semaphoreInfo, nullptr, &imageAvailableSemaphores[i])
				!= vk::Result::eSuccess
			|| device.createSemaphore(&semaphoreInfo, nullptr, &renderFinishedSemaphores[i])
				   != vk::Result::eSuccess
			|| device.createFence(&fenceInfo, nullptr, &inFlightFences[i]) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create semaphores for a frame!");
		}
	}
}

vk::DescriptorPool createDescriptorPool(vk::Device& device, std::vector<vk::Image>& swapchainImages)
{
	vk::DescriptorPoolSize poolSize{.descriptorCount = static_cast<uint32_t>(swapchainImages.size())};

	vk::DescriptorPoolCreateInfo poolInfo{.maxSets       = static_cast<uint32_t>(swapchainImages.size()),
										  .poolSizeCount = 1,
										  .pPoolSizes    = &poolSize};

	vk::DescriptorPool descriptorPool;

	if (device.createDescriptorPool(&poolInfo, nullptr, &descriptorPool) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create descriptor pool!");
	}
	return descriptorPool;
}

void createDescriptorSets(vk::Device& device,
						  vk::DescriptorSetLayout& descriptorSetLayout,
						  std::vector<vk::DescriptorSet>& descriptorSets,
						  std::vector<vk::Image>& swapchainImages,
						  vk::DescriptorPool& descriptorPool,
						  std::vector<vk::Buffer>& uniformBuffers)
{
	std::vector<vk::DescriptorSetLayout> layouts(swapchainImages.size(), descriptorSetLayout);
	vk::DescriptorSetAllocateInfo allocInfo{
		.descriptorPool     = descriptorPool,
		.descriptorSetCount = static_cast<uint32_t>(swapchainImages.size()),
		.pSetLayouts        = layouts.data()};

	descriptorSets.resize(swapchainImages.size());
	if (device.allocateDescriptorSets(&allocInfo, descriptorSets.data()) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}

	for (size_t i = 0; i < swapchainImages.size(); i++) {
		vk::DescriptorBufferInfo bufferInfo{
			.buffer = uniformBuffers[i], .offset = 0, .range = sizeof(UniformBufferObject)};
		vk::WriteDescriptorSet descriptorWrite{.dstSet           = descriptorSets[i],
											   .dstBinding       = 0,
											   .dstArrayElement  = 0,
											   .descriptorCount  = 1,
											   .descriptorType   = vk::DescriptorType::eUniformBuffer,
											   .pImageInfo       = nullptr,
											   .pBufferInfo      = &bufferInfo,
											   .pTexelBufferView = nullptr};
		device.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
	}
}

void updateUniformBuffer(uint32_t currentImage,
						 vk::Device& device,
						 std::vector<vk::DeviceMemory>& uniformBuffersMemory,
						 vk::Extent2D& swapchainExtent)
{
	static auto startTime = std::chrono::high_resolution_clock::now();

	auto currentTime = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

	UniformBufferObject ubo{};
	ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.view =
		glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.proj = glm::perspective(glm::radians(45.0f),
								(float)swapchainExtent.width / (float)swapchainExtent.height, 0.1f, 10.0f);
	ubo.proj[1][1] *= -1;

	void* data;
	device.mapMemory(uniformBuffersMemory[currentImage], 0, sizeof(ubo), {}, &data);
	memcpy(data, &ubo, sizeof(ubo));
	device.unmapMemory(uniformBuffersMemory[currentImage]);
}
