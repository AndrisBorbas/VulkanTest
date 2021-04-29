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

QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface)
{
	QueueFamilyIndices indices;

	uint32_t queueFamilyCount = 0;
	physicalDevice.getQueueFamilyProperties(&queueFamilyCount, nullptr);

	std::vector<vk::QueueFamilyProperties> queueFamilies(queueFamilyCount);
	physicalDevice.getQueueFamilyProperties(&queueFamilyCount, queueFamilies.data());

	int i = 0;
	for (const auto& queueFamily : queueFamilies) {
		if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
			indices.graphicsFamily = i;
		}

		vk::Bool32 presentSupport = false;
		physicalDevice.getSurfaceSupportKHR(i, surface, &presentSupport);
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

vk::SampleCountFlagBits getMaxUsableSampleCount(vk::PhysicalDevice& physicalDevice)
{
	vk::PhysicalDeviceProperties physicalDeviceProperties;
	physicalDevice.getProperties(&physicalDeviceProperties);

	vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts
								  & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
	if (counts & vk::SampleCountFlagBits::e64) return vk::SampleCountFlagBits::e64;
	if (counts & vk::SampleCountFlagBits::e32) return vk::SampleCountFlagBits::e32;
	if (counts & vk::SampleCountFlagBits::e16) return vk::SampleCountFlagBits::e16;
	if (counts & vk::SampleCountFlagBits::e8) return vk::SampleCountFlagBits::e8;
	if (counts & vk::SampleCountFlagBits::e4) return vk::SampleCountFlagBits::e4;
	if (counts & vk::SampleCountFlagBits::e2) return vk::SampleCountFlagBits::e2;

	return vk::SampleCountFlagBits::e1;
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

SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice& physicalDevice,
											  vk::SurfaceKHR& surface)
{
	SwapChainSupportDetails details;

	physicalDevice.getSurfaceCapabilitiesKHR(surface, &details.capabilities);

	uint32_t formatCount;
	physicalDevice.getSurfaceFormatsKHR(surface, &formatCount, nullptr);
	vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);

	if (formatCount != 0) {
		details.formats.resize(formatCount);
		physicalDevice.getSurfaceFormatsKHR(surface, &formatCount, details.formats.data());
	}

	uint32_t presentModeCount;
	physicalDevice.getSurfacePresentModesKHR(surface, &presentModeCount, nullptr);

	if (presentModeCount != 0) {
		details.presentModes.resize(presentModeCount);
		physicalDevice.getSurfacePresentModesKHR(surface, &presentModeCount, details.presentModes.data());
	}

	return details;
}

bool isDeviceSuitable(const vk::PhysicalDevice& physicalDevice,
					  vk::SurfaceKHR& surface,
					  const std::vector<const char*>& deviceExtensions)
{
	if (!findQueueFamilies(physicalDevice, surface).isComplete()) {
		return false;
	}
	if (!checkDeviceExtensionSupport(physicalDevice, deviceExtensions)) {
		return false;
	}

	SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice, surface);
	if (swapChainSupport.formats.empty() || swapChainSupport.presentModes.empty()) {
		return false;
	}

	vk::PhysicalDeviceFeatures2 supportedFeatures;
	physicalDevice.getFeatures2(&supportedFeatures);
	if (!supportedFeatures.features.samplerAnisotropy) {
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
									  const std::vector<const char*>& deviceExtensions,
									  vk::SampleCountFlagBits& msaaSamples)
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

	msaaSamples = getMaxUsableSampleCount(physicalDevice);

	std::cout << "msaa: " << to_string(msaaSamples);

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

	vk::PhysicalDeviceFeatures deviceFeatures{
		.sampleRateShading = VK_TRUE,
		.samplerAnisotropy = VK_TRUE,
	};

	vk::DeviceCreateInfo createInfo{
		.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size()),
		.pQueueCreateInfos       = queueCreateInfos.data(),
		.enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size()),
		.ppEnabledExtensionNames = deviceExtensions.data(),
		.pEnabledFeatures        = &deviceFeatures,
	};

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

	std::cout << std::endl << "Present mode:" << to_string(presentMode) << std::endl;

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

vk::ImageView createImageView(vk::Device& device,
							  vk::Image& image,
							  vk::Format format,
							  vk::ImageAspectFlagBits aspectFlags,
							  uint32_t mipLevels)
{
	vk::ImageViewCreateInfo viewInfo{
		.image    = image,
		.viewType = vk::ImageViewType::e2D,
		.format   = format,
		.subresourceRange =
			{
				.aspectMask     = aspectFlags,
				.baseMipLevel   = 0,
				.levelCount     = mipLevels,
				.baseArrayLayer = 0,
				.layerCount     = 1,
			},
	};

	vk::ImageView imageView;
	if (device.createImageView(&viewInfo, nullptr, &imageView) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create texture image view!");
	}

	return imageView;
}

std::vector<vk::ImageView> createImageViews(vk::Device& device,
											std::vector<vk::Image>& swapchainImages,
											vk::Format& swapchainImageFormat,
											uint32_t mipLevels)
{
	std::vector<vk::ImageView> swapchainImageViews;
	swapchainImageViews.resize(swapchainImages.size());
	for (uint32_t i = 0; i < swapchainImages.size(); i++) {
		swapchainImageViews[i] = createImageView(device, swapchainImages[i], swapchainImageFormat,
												 vk::ImageAspectFlagBits::eColor, mipLevels);
	}
	return swapchainImageViews;
}
