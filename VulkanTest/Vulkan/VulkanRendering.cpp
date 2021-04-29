#include "../Defines.h"

#include "VulkanRendering.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include <stb_image.h>

#include <chrono>

#include "../utils.hpp"
#include "VulkanDevices.hpp"

bool hasStencilComponent(vk::Format format)
{
	return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
}

vk::Format findSupportedFormat(vk::PhysicalDevice& physicalDevice,
							   const std::vector<vk::Format>& candidates,
							   vk::ImageTiling tiling,
							   vk::FormatFeatureFlags features)
{
	for (vk::Format format : candidates) {
		vk::FormatProperties2 properties;
		physicalDevice.getFormatProperties2(format, &properties);
		if (tiling == vk::ImageTiling::eLinear
			&& (properties.formatProperties.linearTilingFeatures & features) == features) {
			return format;
		}
		if (tiling == vk::ImageTiling::eOptimal
			&& (properties.formatProperties.optimalTilingFeatures & features) == features) {
			return format;
		}
	}
	throw std::runtime_error("failed to find supported format!");
}

vk::Format findDepthFormat(vk::PhysicalDevice& physicalDevice)
{
	return findSupportedFormat(
		physicalDevice, {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
		vk::ImageTiling::eOptimal, vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

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
		.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties),
	};

	if (device.allocateMemory(&allocInfo, nullptr, &bufferMemory) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to allocate buffer memory!");
	}

	device.bindBufferMemory(buffer, bufferMemory, 0);
}

vk::CommandBuffer beginSingleTimeCommands(vk::Device& device, vk::CommandPool& commandPool)
{
	vk::CommandBufferAllocateInfo allocInfo{
		.commandPool        = commandPool,
		.level              = vk::CommandBufferLevel::ePrimary,
		.commandBufferCount = 1,
	};

	vk::CommandBuffer commandBuffer;
	device.allocateCommandBuffers(&allocInfo, &commandBuffer);

	vk::CommandBufferBeginInfo beginInfo{
		.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
	};

	commandBuffer.begin(&beginInfo);

	return commandBuffer;
}

void endSingleTimeCommands(vk::Device& device,
						   vk::CommandBuffer& commandBuffer,
						   vk::Queue& graphicsQueue,
						   vk::CommandPool& commandPool)
{
	vkEndCommandBuffer(commandBuffer);

	vk::SubmitInfo submitInfo{
		.commandBufferCount = 1,
		.pCommandBuffers    = &commandBuffer,
	};

	graphicsQueue.submit(1, &submitInfo, {});
	graphicsQueue.waitIdle();

	device.freeCommandBuffers(commandPool, 1, &commandBuffer);
}

void copyBuffer(vk::Device& device,
				vk::CommandPool& commandPool,
				vk::Queue& graphicsQueue,
				vk::Buffer& srcBuffer,
				vk::Buffer& dstBuffer,
				vk::DeviceSize size)
{
	vk::CommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);

	vk::BufferCopy copyRegion{.srcOffset = 0,  // Optional
							  .dstOffset = 0,  // Optional
							  .size      = size};

	commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

	endSingleTimeCommands(device, commandBuffer, graphicsQueue, commandPool);
}

vk::RenderPass createRenderPass(vk::Device& device,
								vk::PhysicalDevice& physicalDevice,
								vk::Format& swapchainImageFormat,
								vk::SampleCountFlagBits msaaSamples)
{
	vk::AttachmentDescription colorAttachment{
		.format         = swapchainImageFormat,
		.samples        = msaaSamples,
		.loadOp         = vk::AttachmentLoadOp::eClear,
		.storeOp        = vk::AttachmentStoreOp::eStore,
		.stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
		.stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
		.initialLayout  = vk::ImageLayout::eUndefined,
		.finalLayout    = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::AttachmentReference colorAttachmentRef{
		.attachment = 0,
		.layout     = vk::ImageLayout::eColorAttachmentOptimal,
	};

	vk::AttachmentDescription depthAttachment{
		.format         = findDepthFormat(physicalDevice),
		.samples        = msaaSamples,
		.loadOp         = vk::AttachmentLoadOp::eClear,
		.storeOp        = vk::AttachmentStoreOp::eDontCare,
		.stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
		.stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
		.initialLayout  = vk::ImageLayout::eUndefined,
		.finalLayout    = vk::ImageLayout::eDepthStencilAttachmentOptimal,
	};
	vk::AttachmentReference depthAttachmentRef{
		.attachment = 1,
		.layout     = vk::ImageLayout::eDepthStencilAttachmentOptimal,
	};

	vk::AttachmentDescription colorAttachmentResolve{
		.format         = swapchainImageFormat,
		.samples        = vk::SampleCountFlagBits::e1,
		.loadOp         = vk::AttachmentLoadOp::eDontCare,
		.storeOp        = vk::AttachmentStoreOp::eStore,
		.stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
		.stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
		.initialLayout  = vk::ImageLayout::eUndefined,
		.finalLayout    = vk::ImageLayout::ePresentSrcKHR,
	};

	vk::AttachmentReference colorAttachmentResolveRef{
		.attachment = 2,
		.layout     = vk::ImageLayout::eColorAttachmentOptimal,
	};

	vk::SubpassDescription subpass{
		.pipelineBindPoint       = vk::PipelineBindPoint::eGraphics,
		.colorAttachmentCount    = 1,
		.pColorAttachments       = &colorAttachmentRef,
		.pResolveAttachments     = &colorAttachmentResolveRef,
		.pDepthStencilAttachment = &depthAttachmentRef,

	};

	vk::SubpassDependency dependency{
		.srcSubpass   = VK_SUBPASS_EXTERNAL,
		.dstSubpass   = 0,
		.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput
						| vk::PipelineStageFlagBits::eEarlyFragmentTests,
		.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput
						| vk::PipelineStageFlagBits::eEarlyFragmentTests,
		.dstAccessMask =
			vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite,
	};

	std::array<vk::AttachmentDescription, 3> attachments = {
		colorAttachment,
		depthAttachment,
		colorAttachmentResolve,
	};

	vk::RenderPassCreateInfo renderPassInfo{
		.attachmentCount = static_cast<uint32_t>(attachments.size()),
		.pAttachments    = attachments.data(),
		.subpassCount    = 1,
		.pSubpasses      = &subpass,
		.dependencyCount = 1,
		.pDependencies   = &dependency,
	};

	vk::RenderPass renderPass;

	if (device.createRenderPass(&renderPassInfo, nullptr, &renderPass) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create render pass!");
	}

	return renderPass;
}

vk::DescriptorSetLayout createDescriptorSetLayout(vk::Device& device)
{
	vk::DescriptorSetLayoutBinding uboLayoutBinding{
		.binding            = 0,
		.descriptorType     = vk::DescriptorType::eUniformBuffer,
		.descriptorCount    = 1,
		.stageFlags         = vk::ShaderStageFlagBits::eVertex,
		.pImmutableSamplers = nullptr,
	};

	vk::DescriptorSetLayoutBinding samplerLayoutBinding{
		.binding            = 1,
		.descriptorType     = vk::DescriptorType::eCombinedImageSampler,
		.descriptorCount    = 1,
		.stageFlags         = vk::ShaderStageFlagBits::eFragment,
		.pImmutableSamplers = nullptr,
	};

	std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};

	vk::DescriptorSetLayoutCreateInfo layoutInfo{
		.bindingCount = static_cast<uint32_t>(bindings.size()),
		.pBindings    = bindings.data(),
	};

	vk::DescriptorSetLayout descriptorSetLayout;

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
									vk::RenderPass& renderPass,
									vk::SampleCountFlagBits msaaSamples)
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

	vk::PipelineRasterizationStateCreateInfo rasterizer{
		.depthClampEnable        = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode             = vk::PolygonMode::eFill,
		.cullMode                = vk::CullModeFlagBits::eBack,
		.frontFace               = vk::FrontFace::eCounterClockwise,
		.depthBiasEnable         = VK_FALSE,
		.depthBiasConstantFactor = 0.0f,
		.depthBiasClamp          = 0.0f,
		.depthBiasSlopeFactor    = 0.0f,
		.lineWidth               = 1.0f,
	};

	vk::PipelineMultisampleStateCreateInfo multisampling{
		.rasterizationSamples  = msaaSamples,
		.sampleShadingEnable   = VK_TRUE,
		.minSampleShading      = 0.2f,
		.pSampleMask           = nullptr,
		.alphaToCoverageEnable = VK_FALSE,
		.alphaToOneEnable      = VK_FALSE,
	};

	vk::PipelineDepthStencilStateCreateInfo depthStencil{
		.depthTestEnable       = VK_TRUE,
		.depthWriteEnable      = VK_TRUE,
		.depthCompareOp        = vk::CompareOp::eLess,
		.depthBoundsTestEnable = VK_FALSE,
		.minDepthBounds        = 0.0f,
		.maxDepthBounds        = 1.0f,
	};

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

	vk::PipelineColorBlendStateCreateInfo colorBlending{
		.logicOpEnable   = VK_FALSE,
		.logicOp         = vk::LogicOp::eCopy,
		.attachmentCount = 1,
		.pAttachments    = &colorBlendAttachment,
	};
	colorBlending.blendConstants[0] = 0.0f;
	colorBlending.blendConstants[1] = 0.0f;
	colorBlending.blendConstants[2] = 0.0f;
	colorBlending.blendConstants[3] = 0.0f;

	vk::DynamicState dynamicStates[] = {vk::DynamicState::eViewport, vk::DynamicState::eLineWidth};

	vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = 2, .pDynamicStates = dynamicStates};

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
		.setLayoutCount         = 1,
		.pSetLayouts            = &descriptorSetLayout,
		.pushConstantRangeCount = 0,
		.pPushConstantRanges    = nullptr,
	};

	if (device.createPipelineLayout(&pipelineLayoutInfo, nullptr, &pipelineLayout) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create pipeline layout!");
	}

	vk::GraphicsPipelineCreateInfo pipelineInfo{
		.stageCount          = 2,
		.pStages             = shaderStages,
		.pVertexInputState   = &vertexInputInfo,
		.pInputAssemblyState = &inputAssembly,
		.pViewportState      = &viewportState,
		.pRasterizationState = &rasterizer,
		.pMultisampleState   = &multisampling,
		.pDepthStencilState  = &depthStencil,
		.pColorBlendState    = &colorBlending,
		.pDynamicState       = nullptr,
		.layout              = pipelineLayout,
		.renderPass          = renderPass,
		.subpass             = 0,
		.basePipelineHandle  = nullptr,
		.basePipelineIndex   = -1,
	};

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
						std::vector<Vertex>& vertices,
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
					   std::vector<uint32_t>& indices,
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
						vk::ImageView& colorImageView,
						vk::ImageView& depthImageView,
						vk::RenderPass& renderPass,
						vk::Extent2D& swapchainExtent)
{
	swapChainFramebuffers.resize(swapchainImageViews.size());
	for (size_t i = 0; i < swapchainImageViews.size(); i++) {
		std::array<vk::ImageView, 3> attachments = {colorImageView, depthImageView, swapchainImageViews[i]};

		vk::FramebufferCreateInfo framebufferInfo{
			.renderPass      = renderPass,
			.attachmentCount = static_cast<uint32_t>(attachments.size()),
			.pAttachments    = attachments.data(),
			.width           = swapchainExtent.width,
			.height          = swapchainExtent.height,
			.layers          = 1,
		};
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
						  std::vector<uint32_t>& indices)
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

		std::array<vk::ClearValue, 2> clearValues{};
		clearValues[0].setColor(std::array<float, 4>{0.01f, 0.01f, 0.0125f, 0.0f});
		clearValues[1].setDepthStencil({1.0f, 0});

		vk::RenderPassBeginInfo renderPassInfo{.renderPass      = renderPass,
											   .framebuffer     = swapChainFramebuffers[i],
											   .clearValueCount = static_cast<uint32_t>(clearValues.size()),
											   .pClearValues    = clearValues.data()};

		renderPassInfo.renderArea.offset = VkOffset2D{0, 0};
		renderPassInfo.renderArea.extent = swapchainExtent;

		commandBuffers[i].beginRenderPass(&renderPassInfo, vk::SubpassContents::eInline);

		commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

		vk::Buffer vertexBuffers[] = {vertexBuffer};
		vk::DeviceSize offsets[]   = {0};
		commandBuffers[i].bindVertexBuffers(0, 1, vertexBuffers, offsets);
		commandBuffers[i].bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);

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
	std::array<vk::DescriptorPoolSize, 2> poolSizes{};
	poolSizes[0].type            = vk::DescriptorType::eUniformBuffer;
	poolSizes[0].descriptorCount = static_cast<uint32_t>(swapchainImages.size());
	poolSizes[1].type            = vk::DescriptorType::eCombinedImageSampler;
	poolSizes[1].descriptorCount = static_cast<uint32_t>(swapchainImages.size());

	vk::DescriptorPoolCreateInfo poolInfo{
		.maxSets       = static_cast<uint32_t>(swapchainImages.size()),
		.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
		.pPoolSizes    = poolSizes.data(),
	};

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
						  std::vector<vk::Buffer>& uniformBuffers,
						  vk::ImageView& textureImageView,
						  vk::Sampler& textureSampler)
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
			.buffer = uniformBuffers[i],
			.offset = 0,
			.range  = sizeof(UniformBufferObject),
		};
		vk::DescriptorImageInfo imageInfo{
			.sampler     = textureSampler,
			.imageView   = textureImageView,
			.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		};

		std::array<vk::WriteDescriptorSet, 2> descriptorWrites{};

		descriptorWrites[0].dstSet          = descriptorSets[i];
		descriptorWrites[0].dstBinding      = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType  = vk::DescriptorType::eUniformBuffer;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo     = &bufferInfo;

		descriptorWrites[1].dstSet          = descriptorSets[i];
		descriptorWrites[1].dstBinding      = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pImageInfo      = &imageInfo;

		device.updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(),
									0, nullptr);
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
	ubo.model = glm::rotate(glm::mat4(1.0f), glm::sin(time * glm::radians(90.0f)) * 0.45f,
							glm::vec3(0.0f, 0.0f, 1.0f));
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

void copyBufferToImage(vk::Device& device,
					   vk::Queue& graphicsQueue,
					   vk::CommandPool& commandPool,
					   vk::Buffer buffer,
					   vk::Image image,
					   uint32_t width,
					   uint32_t height)
{
	vk::CommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);

	vk::BufferImageCopy region{
		.bufferOffset      = 0,
		.bufferRowLength   = 0,
		.bufferImageHeight = 0,
		.imageSubresource =
			{
				.aspectMask     = vk::ImageAspectFlagBits::eColor,
				.mipLevel       = 0,
				.baseArrayLayer = 0,
				.layerCount     = 1,
			},
		.imageOffset = {0, 0, 0},
		.imageExtent = {width, height, 1},
	};

	commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

	endSingleTimeCommands(device, commandBuffer, graphicsQueue, commandPool);
}

void transitionImageLayout(vk::Device& device,
						   vk::Queue& graphicsQueue,
						   vk::CommandPool& commandPool,
						   vk::Image image,
						   vk::Format format,
						   vk::ImageLayout oldLayout,
						   vk::ImageLayout newLayout,
						   uint32_t mipLevels)
{
	vk::CommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);

	vk::ImageMemoryBarrier barrier{
		.oldLayout           = oldLayout,
		.newLayout           = newLayout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image               = image,
		.subresourceRange =
			{
				.baseMipLevel   = 0,
				.levelCount     = mipLevels,
				.baseArrayLayer = 0,
				.layerCount     = 1,
			},
	};

	vk::PipelineStageFlags sourceStage;
	vk::PipelineStageFlags destinationStage;

	if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
		barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;

		if (hasStencilComponent(format)) {
			barrier.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
		}
	} else {
		barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
	}

	if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
		barrier.srcAccessMask = {};
		barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
		sourceStage           = vk::PipelineStageFlagBits::eTopOfPipe;
		destinationStage      = vk::PipelineStageFlagBits::eTransfer;
	} else if (oldLayout == vk::ImageLayout::eTransferDstOptimal
			   && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
		sourceStage           = vk::PipelineStageFlagBits::eTransfer;
		destinationStage      = vk::PipelineStageFlagBits::eFragmentShader;
	} else if (oldLayout == vk::ImageLayout::eUndefined
			   && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
		barrier.srcAccessMask = {};
		barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead
								| vk::AccessFlagBits::eDepthStencilAttachmentWrite;
		sourceStage      = vk::PipelineStageFlagBits::eTopOfPipe;
		destinationStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
	} else {
		throw std::invalid_argument("unsupported layout transition!");
	}

	commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, 0, nullptr, 0, nullptr, 1, &barrier);

	endSingleTimeCommands(device, commandBuffer, graphicsQueue, commandPool);
}

std::tuple<vk::Image, vk::DeviceMemory> createImage(vk::Device& device,
													vk::PhysicalDevice& physicalDevice,
													uint32_t width,
													uint32_t height,
													uint32_t mipLevels,
													vk::SampleCountFlagBits sampleNum,
													vk::Format format,
													vk::ImageTiling tiling,
													vk::ImageUsageFlags usage,
													vk::MemoryPropertyFlags properties)
{
	vk::ImageCreateInfo imageInfo{
		.flags     = {},
		.imageType = vk::ImageType::e2D,
		.format    = format,
		.extent =
			{
				.width  = static_cast<uint32_t>(width),
				.height = static_cast<uint32_t>(height),
				.depth  = 1,
			},
		.mipLevels     = mipLevels,
		.arrayLayers   = 1,
		.samples       = sampleNum,
		.tiling        = tiling,
		.usage         = usage,
		.sharingMode   = vk::SharingMode::eExclusive,
		.initialLayout = vk::ImageLayout::eUndefined,
	};

	vk::Image image;

	if (device.createImage(&imageInfo, nullptr, &image) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create image!");
	}

	vk::MemoryRequirements memRequirements;
	device.getImageMemoryRequirements(image, &memRequirements);

	vk::MemoryAllocateInfo allocInfo{
		.allocationSize  = memRequirements.size,
		.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties)};

	vk::DeviceMemory imageMemory;

	if (device.allocateMemory(&allocInfo, nullptr, &imageMemory) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to allocate image memory!");
	}

	device.bindImageMemory(image, imageMemory, 0);

	return std::make_tuple(image, imageMemory);
}

void generateMipmaps(vk::Device& device,
					 vk::PhysicalDevice& physicalDevice,
					 vk::CommandPool& commandPool,
					 vk::Queue& graphicsQueue,
					 vk::Image& image,
					 vk::Format format,
					 int32_t texWidth,
					 int32_t texHeight,
					 uint32_t mipLevels)
{
	vk::FormatProperties formatProperties;
	physicalDevice.getFormatProperties(format, &formatProperties);

	if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
		throw std::runtime_error("texture image format does not support linear blitting!");
	}

	vk::CommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);

	vk::ImageMemoryBarrier barrier{
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image               = image,
		.subresourceRange =
			{
				.aspectMask     = vk::ImageAspectFlagBits::eColor,
				.levelCount     = 1,
				.baseArrayLayer = 0,
				.layerCount     = 1,

			},
	};

	int32_t mipWidth  = texWidth;
	int32_t mipHeight = texHeight;

	for (uint32_t i = 1; i < mipLevels; i++) {
		barrier.subresourceRange.baseMipLevel = i - 1;
		barrier.oldLayout                     = vk::ImageLayout::eTransferDstOptimal;
		barrier.newLayout                     = vk::ImageLayout::eTransferSrcOptimal;
		barrier.srcAccessMask                 = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask                 = vk::AccessFlagBits::eTransferRead;

		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
									  vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 0, nullptr, 1,
									  &barrier);

		vk::ImageBlit blit{
			.srcSubresource =
				{
					.aspectMask     = vk::ImageAspectFlagBits::eColor,
					.mipLevel       = i - 1,
					.baseArrayLayer = 0,
					.layerCount     = 1,
				},
			.srcOffsets = vk::ArrayWrapper1D<vk::Offset3D, 2>{{vk::Offset3D{0, 0, 0},
															   vk::Offset3D{mipWidth, mipHeight, 1}}},
			.dstSubresource =
				{
					.aspectMask     = vk::ImageAspectFlagBits::eColor,
					.mipLevel       = i,
					.baseArrayLayer = 0,
					.layerCount     = 1,
				},
			.dstOffsets =
				vk::ArrayWrapper1D<vk::Offset3D, 2>{
					{vk::Offset3D{0, 0, 0},
					 vk::Offset3D{mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1}}},
		};

		commandBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image,
								vk::ImageLayout::eTransferDstOptimal, 1, &blit, vk::Filter::eLinear);

		barrier.oldLayout     = vk::ImageLayout::eTransferSrcOptimal;
		barrier.newLayout     = vk::ImageLayout::eShaderReadOnlyOptimal;
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
		barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
									  vk::PipelineStageFlagBits::eFragmentShader, {}, 0, nullptr, 0, nullptr,
									  1, &barrier);

		if (mipWidth > 1) {
			mipWidth /= 2;
		}
		if (mipHeight > 1) {
			mipHeight /= 2;
		}
	}

	barrier.subresourceRange.baseMipLevel = mipLevels - 1;
	barrier.oldLayout                     = vk::ImageLayout::eTransferDstOptimal;
	barrier.newLayout                     = vk::ImageLayout::eShaderReadOnlyOptimal;
	barrier.srcAccessMask                 = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask                 = vk::AccessFlagBits::eShaderRead;

	commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
								  vk::PipelineStageFlagBits::eFragmentShader, {}, 0, nullptr, 0, nullptr, 1,
								  &barrier);

	endSingleTimeCommands(device, commandBuffer, graphicsQueue, commandPool);
}

std::tuple<vk::Image, vk::DeviceMemory> createTextureImage(vk::Device& device,
														   vk::PhysicalDevice& physicalDevice,
														   const char* filename,
														   int stbiChannels,
														   uint32_t& mipLevels,
														   vk::Format format,
														   vk::Queue& graphicsQueue,
														   vk::CommandPool& commandPool)
{
	int texWidth, texHeight, texChannels;
	stbi_uc* pixels = stbi_load(filename, &texWidth, &texHeight, &texChannels, stbiChannels);
	mipLevels       = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;
	vk::DeviceSize imageSize = texWidth * texHeight * 4;

	if (!pixels) {
		throw std::runtime_error("failed to load texture image!");
	}

	vk::Buffer stagingBuffer;
	vk::DeviceMemory stagingBufferMemory;

	createBuffer(device, physicalDevice, imageSize, vk::BufferUsageFlagBits::eTransferSrc,
				 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
				 stagingBuffer, stagingBufferMemory);

	void* data;
	device.mapMemory(stagingBufferMemory, 0, imageSize, {}, &data);
	memcpy(data, pixels, static_cast<size_t>(imageSize));
	device.unmapMemory(stagingBufferMemory);

	stbi_image_free(pixels);

	vk::Image textureImage;
	vk::DeviceMemory textureImageMemory;

	std::tie(textureImage, textureImageMemory) =
		createImage(device, physicalDevice, texWidth, texHeight, mipLevels, vk::SampleCountFlagBits::e1,
					format, vk::ImageTiling::eOptimal,
					vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst
						| vk::ImageUsageFlagBits::eSampled,
					vk::MemoryPropertyFlagBits::eDeviceLocal);

	transitionImageLayout(device, graphicsQueue, commandPool, textureImage, format,
						  vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels);
	copyBufferToImage(device, graphicsQueue, commandPool, stagingBuffer, textureImage,
					  static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
	//	transitionImageLayout(device, graphicsQueue, commandPool, textureImage, format,
	//						  vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
	//						  mipLevels);

	generateMipmaps(device, physicalDevice, commandPool, graphicsQueue, textureImage,
					vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);

	device.destroyBuffer(stagingBuffer, nullptr);
	device.freeMemory(stagingBufferMemory, nullptr);

	return std::tie(textureImage, textureImageMemory);
}

vk::Sampler createTextureSampler(vk::Device& device, vk::PhysicalDevice& physicalDevice, uint32_t mipLevels)
{
	vk::PhysicalDeviceProperties properties{};
	physicalDevice.getProperties(&properties);

	vk::SamplerCreateInfo samplerInfo{
		.magFilter               = vk::Filter::eLinear,
		.minFilter               = vk::Filter::eLinear,
		.mipmapMode              = vk::SamplerMipmapMode::eLinear,
		.addressModeU            = vk::SamplerAddressMode::eRepeat,
		.addressModeV            = vk::SamplerAddressMode::eRepeat,
		.addressModeW            = vk::SamplerAddressMode::eRepeat,
		.mipLodBias              = 0.0f,
		.anisotropyEnable        = VK_TRUE,
		.maxAnisotropy           = properties.limits.maxSamplerAnisotropy,
		.compareEnable           = VK_FALSE,
		.compareOp               = vk::CompareOp::eAlways,
		.minLod                  = 0.0f,
		.maxLod                  = static_cast<float>(mipLevels),
		.borderColor             = vk::BorderColor::eIntTransparentBlack,
		.unnormalizedCoordinates = VK_FALSE,
	};

	vk::Sampler textureSampler;

	if (device.createSampler(&samplerInfo, nullptr, &textureSampler) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create texture sampler!");
	}

	return textureSampler;
}

std::tuple<vk::Image, vk::DeviceMemory, vk::ImageView> createDepthResources(
	vk::Device& device,
	vk::PhysicalDevice& physicalDevice,
	vk::Extent2D& swapchainExtent,
	vk::Queue& graphicsQueue,
	vk::CommandPool& commandPool,
	vk::SampleCountFlagBits msaaSamples)
{
	vk::Format depthFormat = findDepthFormat(physicalDevice);

	vk::Image depthImage;
	vk::DeviceMemory depthDeviceMemory;

	std::tie(depthImage, depthDeviceMemory) =
		createImage(device, physicalDevice, swapchainExtent.width, swapchainExtent.height, 1, msaaSamples,
					depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment,
					vk::MemoryPropertyFlagBits::eDeviceLocal);
	vk::ImageView depthImageView =
		createImageView(device, depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);

	transitionImageLayout(device, graphicsQueue, commandPool, depthImage, depthFormat,
						  vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, 1);

	return std::tie(depthImage, depthDeviceMemory, depthImageView);
}

std::tuple<vk::Image, vk::DeviceMemory, vk::ImageView> createColorResources(
	vk::Device& device,
	vk::PhysicalDevice& physicalDevice,
	vk::Format& swapchainImageFormat,
	vk::Extent2D& swapchainExtent,
	vk::SampleCountFlagBits msaaSamples)
{
	vk::Format colorFormat = swapchainImageFormat;

	vk::Image colorImage;
	vk::DeviceMemory colorImageMemory;
	vk::ImageView colorImageView;

	std::tie(colorImage, colorImageMemory) =
		createImage(device, physicalDevice, swapchainExtent.width, swapchainExtent.height, 1, msaaSamples,
					colorFormat, vk::ImageTiling::eOptimal,
					vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
					vk::MemoryPropertyFlagBits::eDeviceLocal);
	colorImageView = createImageView(device, colorImage, colorFormat, vk::ImageAspectFlagBits::eColor, 1);

	return std::tie(colorImage, colorImageMemory, colorImageView);
}
