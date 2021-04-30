#pragma once

#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>

#include <optional>
#include <set>
#include <tuple>
#include <vector>

#include "../Shaders.hpp"

vk::ShaderModule createShaderModule(vk::Device device, const std::vector<char>& code);

vk::RenderPass createRenderPass(vk::Device& device,
								vk::PhysicalDevice& physicalDevice,
								vk::Format& swapchainImageFormat,
								vk::SampleCountFlagBits msaaSamples);

vk::DescriptorSetLayout createDescriptorSetLayout(vk::Device& device);

vk::Pipeline createGraphicsPipeline(vk::Device& device,
									vk::Extent2D& swapchainExtent,
									vk::DescriptorSetLayout& descriptorSetLayout,
									vk::PipelineLayout& pipelineLayout,
									vk::RenderPass& renderPass,
									vk::SampleCountFlagBits msaaSamples);

vk::CommandBuffer beginSingleTimeCommands(vk::Device& device, vk::CommandPool& commandPool);
void endSingleTimeCommands(vk::Device& device,
						   vk::CommandBuffer& commandBuffer,
						   vk::Queue& graphicsQueue,
						   vk::CommandPool& commandPool);

void createVertexBuffer(vk::Device& device,
						vk::PhysicalDevice& physicalDevice,
						vk::CommandPool& commandPool,
						vk::Queue& graphicsQueue,
						std::vector<Vertex>& vertices,
						vk::Buffer& vertexBuffer,
						vk::DeviceMemory& vertexBufferMemory);

void createIndexBuffer(vk::Device& device,
					   vk::PhysicalDevice& physicalDevice,
					   vk::CommandPool& commandPool,
					   vk::Queue& graphicsQueue,
					   std::vector<uint32_t>& indices,
					   vk::Buffer& indexBuffer,
					   vk::DeviceMemory& indexBufferMemory);

void createUniformBuffers(vk::Device& device,
						  vk::PhysicalDevice& physicalDevice,
						  std::vector<vk::Buffer>& uniformBuffers,
						  std::vector<vk::DeviceMemory>& uniformBuffersMemory,
						  std::vector<vk::Image>& swapchainImages);

void createFramebuffers(vk::Device& device,
						std::vector<vk::Framebuffer>& swapChainFramebuffers,
						std::vector<vk::ImageView>& swapchainImageViews,
						vk::ImageView& colorImageView,
						vk::ImageView& depthImageView,
						vk::RenderPass& renderPass,
						vk::Extent2D& swapchainExtent);

vk::CommandPool createCommandPool(vk::Device& device,
								  vk::PhysicalDevice& physicalDevice,
								  vk::SurfaceKHR& surface);

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
						  std::vector<uint32_t>& indices);

void createSyncObjects(vk::Device& device,
					   std::vector<vk::Semaphore>& imageAvailableSemaphores,
					   std::vector<vk::Semaphore>& renderFinishedSemaphores,
					   std::vector<vk::Fence>& inFlightFences,
					   std::vector<vk::Fence>& imagesInFlight,
					   const int MAX_FRAMES_IN_FLIGHT,
					   std::vector<vk::Image>& swapchainImages);

vk::DescriptorPool createDescriptorPool(vk::Device& device, std::vector<vk::Image>& swapchainImages);

void createDescriptorSets(vk::Device& device,
						  vk::DescriptorSetLayout& descriptorSetLayout,
						  std::vector<vk::DescriptorSet>& descriptorSets,
						  std::vector<vk::Image>& swapchainImages,
						  vk::DescriptorPool& descriptorPool,
						  std::vector<vk::Buffer>& uniformBuffers,
						  vk::ImageView& textureImageView,
						  vk::Sampler& textureSampler);

void updateUniformBuffer(uint32_t currentImage,
						 vk::Device& device,
						 std::vector<vk::DeviceMemory>& uniformBuffersMemory,
						 vk::Extent2D& swapchainExtent);

std::tuple<vk::Image, vk::DeviceMemory> createTextureImage(vk::Device& device,
														   vk::PhysicalDevice& physicalDevice,
														   const char* filename,
														   int stbiChannels,
														   uint32_t& mipLevels,
														   vk::Format format,
														   vk::Queue& graphicsQueue,
														   vk::CommandPool& commandPool);

vk::Sampler createTextureSampler(vk::Device& device, vk::PhysicalDevice& physicalDevice, uint32_t mipLevels);

std::tuple<vk::Image, vk::DeviceMemory, vk::ImageView> createDepthResources(
	vk::Device& device,
	vk::PhysicalDevice& physicalDevice,
	vk::Extent2D& swapchainExtent,
	vk::Queue& graphicsQueue,
	vk::CommandPool& commandPool,
	vk::SampleCountFlagBits msaaSamples);

std::tuple<vk::Image, vk::DeviceMemory, vk::ImageView> createColorResources(
	vk::Device& device,
	vk::PhysicalDevice& physicalDevice,
	vk::Format& swapchainImageFormat,
	vk::Extent2D& swapchainExtent,
	vk::SampleCountFlagBits msaaSamples);
