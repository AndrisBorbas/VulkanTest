#pragma once

#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>

#include <optional>
#include <set>
#include <vector>

#include "../Shaders.hpp"

vk::ShaderModule createShaderModule(vk::Device device, const std::vector<char>& code);

vk::RenderPass createRenderPass(vk::Device& device, vk::Format& swapchainImageFormat);

vk::DescriptorSetLayout createDescriptorSetLayout(vk::Device& device);
vk::Pipeline createGraphicsPipeline(vk::Device& device,

									vk::Extent2D& swapchainExtent,
									vk::DescriptorSetLayout& descriptorSetLayout,
									vk::PipelineLayout& pipelineLayout,
									vk::RenderPass& renderPass);

void createVertexBuffer(vk::Device& device,
						vk::PhysicalDevice& physicalDevice,
						vk::CommandPool& commandPool,
						vk::Queue& graphicsQueue,
						const std::vector<Vertex>& vertices,
						vk::Buffer& vertexBuffer,
						vk::DeviceMemory& vertexBufferMemory);

void createIndexBuffer(vk::Device& device,
					   vk::PhysicalDevice& physicalDevice,
					   vk::CommandPool& commandPool,
					   vk::Queue& graphicsQueue,
					   const std::vector<uint16_t>& indices,
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
						  const std::vector<uint16_t>& indices);

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
						  std::vector<vk::Buffer>& uniformBuffers);

void updateUniformBuffer(uint32_t currentImage,
						 vk::Device& device,
						 std::vector<vk::DeviceMemory>& uniformBuffersMemory,
						 vk::Extent2D& swapchainExtent);
