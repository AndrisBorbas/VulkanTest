#pragma once

#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>

#include <optional>
#include <set>
#include <vector>

vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes,
										 vk::PresentModeKHR preferredPresentMode)
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

vk::ShaderModule createShaderModule(vk::Device device, const std::vector<char>& code)
{
	vk::ShaderModuleCreateInfo createInfo{.codeSize = code.size(),
										  .pCode    = reinterpret_cast<const uint32_t*>(code.data())};
	vk::ShaderModule shaderModule;
	if (device.createShaderModule(&createInfo, nullptr, &shaderModule) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create shader module!");
	}
	return shaderModule;
}
