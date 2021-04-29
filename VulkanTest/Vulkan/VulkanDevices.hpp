#pragma once
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>

#include <optional>
#include <set>
#include <vector>

struct QueueFamilyIndices
{
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete() { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

struct SwapChainSupportDetails
{
	vk::SurfaceCapabilitiesKHR capabilities;
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR> presentModes;
};

QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface);

vk::PhysicalDevice pickPhysicalDevice(vk::Instance& instance,
									  vk::SurfaceKHR& surface,
									  const std::vector<const char*>& deviceExtensions,
									  vk::SampleCountFlagBits& msaaSamples);

vk::Device createLogicalDevice(vk::Instance& instance,
							   vk::SurfaceKHR& surface,
							   vk::PhysicalDevice& physicalDevice,
							   vk::Queue& graphicsQueue,
							   vk::Queue& presentQueue,
							   const std::vector<const char*>& deviceExtensions,
							   const std::vector<const char*>& validationLayers);

vk::SwapchainKHR createSwapChain(vk::Device& device,
								 vk::PhysicalDevice& physicalDevice,
								 GLFWwindow* window,
								 vk::SurfaceKHR& surface,
								 std::vector<vk::Image>& swapchainImages,
								 vk::Format& swapchainImageFormat,
								 vk::Extent2D& swapchainExtent,
								 const vk::PresentModeKHR& preferredPresentMode);

vk::ImageView createImageView(vk::Device& device,
							  vk::Image& image,
							  vk::Format format,
							  vk::ImageAspectFlagBits aspectFlags,
							  uint32_t mipLevels);

std::vector<vk::ImageView> createImageViews(vk::Device& device,
											std::vector<vk::Image>& swapchainImages,
											vk::Format& swapchainImageFormat,
											uint32_t mipLevels);
