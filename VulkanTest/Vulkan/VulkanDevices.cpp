#include "../Defines.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

#include "VulkanDevices.hpp"

QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice device, vk::SurfaceKHR surface)
{
	QueueFamilyIndices indices;

	uint32_t queueFamilyCount = 0;
	device.getQueueFamilyProperties(&queueFamilyCount, nullptr);

	std::vector<vk::QueueFamilyProperties> queueFamilies(queueFamilyCount);
	device.getQueueFamilyProperties(&queueFamilyCount, queueFamilies.data());

	int i = 0;
	for (const auto& queueFamily : queueFamilies) {
		if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
			indices.graphicsFamily = i;
		}

		vk::Bool32 presentSupport = false;
		device.getSurfaceSupportKHR(i, surface, &presentSupport);
		if (presentSupport) {
			indices.presentFamily = i;
		}

		// early exit
		if (indices.isComplete()) {
			break;
		}

		i++;
	}

	return indices;
}

bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device,
								 const std::vector<const char*>& deviceExtensions)
{
	uint32_t extensionCount;
	device.enumerateDeviceExtensionProperties(nullptr, &extensionCount, nullptr);

	std::vector<vk::ExtensionProperties> availableExtensions(extensionCount);
	device.enumerateDeviceExtensionProperties(nullptr, &extensionCount, availableExtensions.data());

	std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

	for (const auto& extension : availableExtensions) {
		requiredExtensions.erase(extension.extensionName);
	}

	return requiredExtensions.empty();
}

SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice& device, vk::SurfaceKHR& surface)
{
	SwapChainSupportDetails details;

	device.getSurfaceCapabilitiesKHR(surface, &details.capabilities);

	uint32_t formatCount;
	device.getSurfaceFormatsKHR(surface, &formatCount, nullptr);
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

	if (formatCount != 0) {
		details.formats.resize(formatCount);
		device.getSurfaceFormatsKHR(surface, &formatCount, details.formats.data());
	}

	uint32_t presentModeCount;
	device.getSurfacePresentModesKHR(surface, &presentModeCount, nullptr);

	if (presentModeCount != 0) {
		details.presentModes.resize(presentModeCount);
		device.getSurfacePresentModesKHR(surface, &presentModeCount, details.presentModes.data());
	}

	return details;
}

bool isDeviceSuitable(const vk::PhysicalDevice& device,
					  vk::SurfaceKHR& surface,
					  const std::vector<const char*>& deviceExtensions)
{
	if (!findQueueFamilies(device, surface).isComplete()) {
		return false;
	}
	if (!checkDeviceExtensionSupport(device, deviceExtensions)) {
		return false;
	}

	SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, surface);
	if (swapChainSupport.formats.empty() || swapChainSupport.presentModes.empty()) {
		return false;
	}

	return true;
}

int rateDeviceSuitability(const vk::PhysicalDevice& device,
						  vk::SurfaceKHR& surface,
						  const std::vector<const char*>& deviceExtensions)
{
	if (!isDeviceSuitable(device, surface, deviceExtensions)) {
		return -1;
	}

	vk::PhysicalDeviceProperties deviceProperties;
	device.getProperties(&deviceProperties);

	vk::PhysicalDeviceFeatures deviceFeatures;
	device.getFeatures(&deviceFeatures);

	// Application can't function without geometry shaders
	if (!deviceFeatures.geometryShader) {
		return 0;
	}

	int score = 0;

	// Discrete GPUs have a significant performance advantage
	if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
		score += 1000;
	}

	// Maximum possible size of textures affects graphics quality
	score += deviceProperties.limits.maxImageDimension2D;

	return score;
}

vk::PhysicalDevice pickPhysicalDevice(vk::Instance& instance,
									  vk::SurfaceKHR& surface,
									  const std::vector<const char*>& deviceExtensions)
{
	uint32_t deviceCount = 0;
	instance.enumeratePhysicalDevices(&deviceCount, nullptr);
	if (deviceCount == 0) {
		throw std::runtime_error("failed to find GPUs with Vulkan support!");
	}
	std::vector<vk::PhysicalDevice> devices(deviceCount);
	instance.enumeratePhysicalDevices(&deviceCount, devices.data());

	// Use an ordered map to automatically sort candidates by increasing score
	std::multimap<int, VkPhysicalDevice> candidates;

	for (const auto& item : devices) {
		int score = rateDeviceSuitability(item, surface, deviceExtensions);
		candidates.insert(std::make_pair(score, item));
	}

	vk::PhysicalDevice physicalDevice;

	// Check if the best candidate is suitable at all
	if (candidates.rbegin()->first > 0) {
		physicalDevice = candidates.rbegin()->second;
		std::cout << physicalDevice.getProperties().deviceName << std::endl;
	} else {
		throw std::runtime_error("failed to find a suitable GPU!");
	}

	return physicalDevice;
}

vk::Device createLogicalDevice(vk::Instance& instance,
							   vk::SurfaceKHR& surface,
							   vk::PhysicalDevice& physicalDevice,
							   vk::Queue& graphicsQueue,
							   vk::Queue& presentQueue,
							   const std::vector<const char*>& deviceExtensions,
							   const std::vector<const char*>& validationLayers)
{
	QueueFamilyIndices queueFamilies = findQueueFamilies(physicalDevice, surface);

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

	vk::DeviceCreateInfo createInfo{.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size()),
									.pQueueCreateInfos       = queueCreateInfos.data(),
									.enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size()),
									.ppEnabledExtensionNames = deviceExtensions.data(),
									.pEnabledFeatures        = &deviceFeatures};

#ifdef ENABLE_VALIDATION_LAYERS
	createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
	createInfo.ppEnabledLayerNames = validationLayers.data();
#else
	createInfo.enabledLayerCount = 0;
#endif

	vk::Device device;

	if (physicalDevice.createDevice(&createInfo, nullptr, &device) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create logical device!");
	}

	device.getQueue(queueFamilies.graphicsFamily.value(), 0, &graphicsQueue);
	device.getQueue(queueFamilies.presentFamily.value(), 0, &presentQueue);

	return device;
}

vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes,
										 const vk::PresentModeKHR& preferredPresentMode)
{
	// Preferred
	if (std::any_of(availablePresentModes.cbegin(), availablePresentModes.cend(),
					[&](auto a) { return (a == preferredPresentMode); })) {
		return preferredPresentMode;
	}

	// Fallback
	if (std::any_of(availablePresentModes.cbegin(), availablePresentModes.cend(),
					[](auto a) { return (a == vk::PresentModeKHR::eMailbox); })) {
		return vk::PresentModeKHR::eMailbox;
	}

	// Default
	return vk::PresentModeKHR::eFifo;
}

vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, GLFWwindow* window)
{
	if (capabilities.currentExtent.width != UINT32_MAX) {
		return capabilities.currentExtent;
	} else {
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		vk::Extent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

		actualExtent.width  = std::max(capabilities.minImageExtent.width,
                                      std::min(capabilities.maxImageExtent.width, actualExtent.width));
		actualExtent.height = std::max(capabilities.minImageExtent.height,
									   std::min(capabilities.maxImageExtent.height, actualExtent.height));

		return actualExtent;
	}
}

vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
	for (const auto& availableFormat : availableFormats) {
		if (availableFormat.format == vk::Format::eB8G8R8A8Srgb
			&& availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
			return availableFormat;
		}
	}

	// Fallback
	return availableFormats[0];
}

vk::SwapchainKHR createSwapChain(vk::Device& device,
								 vk::PhysicalDevice& physicalDevice,
								 GLFWwindow* window,
								 vk::SurfaceKHR& surface,
								 std::vector<vk::Image>& swapchainImages,
								 vk::Format& swapchainImageFormat,
								 vk::Extent2D& swapchainExtent,
								 const vk::PresentModeKHR& preferredPresentMode)
{
	const SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice, surface);

	const vk::PresentModeKHR presentMode =
		chooseSwapPresentMode(swapChainSupport.presentModes, preferredPresentMode);
	const vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
	const vk::Extent2D extent                = chooseSwapExtent(swapChainSupport.capabilities, window);

	std::cout << std::endl << to_string(presentMode) << std::endl;

	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

	if (swapChainSupport.capabilities.maxImageCount > 0
		&& imageCount > swapChainSupport.capabilities.maxImageCount) {
		imageCount = swapChainSupport.capabilities.maxImageCount;
	}

	vk::SwapchainCreateInfoKHR createInfo{.surface          = surface,
										  .minImageCount    = imageCount,
										  .imageFormat      = surfaceFormat.format,
										  .imageColorSpace  = surfaceFormat.colorSpace,
										  .imageExtent      = extent,
										  .imageArrayLayers = 1,
										  .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment};

	QueueFamilyIndices queueFamilies = findQueueFamilies(physicalDevice, surface);
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
	createInfo.presentMode    = presentMode;
	createInfo.clipped        = true;
	createInfo.oldSwapchain   = nullptr;

	vk::SwapchainKHR swapchain;

	if (device.createSwapchainKHR(&createInfo, nullptr, &swapchain) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create swap chain!");
	}

	swapchainImageFormat = surfaceFormat.format;
	swapchainExtent      = extent;

	device.getSwapchainImagesKHR(swapchain, &imageCount, nullptr);
	swapchainImages.resize(imageCount);
	device.getSwapchainImagesKHR(swapchain, &imageCount, swapchainImages.data());

	return swapchain;
}

std::vector<vk::ImageView> createImageViews(vk::Device& device,
											std::vector<vk::Image>& swapchainImages,
											vk::Format& swapchainImageFormat)
{
	std::vector<vk::ImageView> swapchainImageViews;
	swapchainImageViews.resize(swapchainImages.size());
	for (size_t i = 0; i < swapchainImages.size(); i++) {
		vk::ImageViewCreateInfo createInfo{
			.image    = swapchainImages[i],
			.viewType = vk::ImageViewType::e2D,
			.format   = swapchainImageFormat,
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

		if (device.createImageView(&createInfo, nullptr, &swapchainImageViews[i]) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create image views!");
		}
	}
	return swapchainImageViews;
}
