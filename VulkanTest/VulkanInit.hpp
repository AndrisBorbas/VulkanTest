#pragma once

#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>

#include <optional>
#include <vector>

bool checkValidationLayerSupport(std::vector<const char*> validationLayers)
{
	uint32_t layerCount;
	vk::enumerateInstanceLayerProperties(&layerCount, nullptr);

	std::vector<vk::LayerProperties> availableLayers(layerCount);
	vk::enumerateInstanceLayerProperties(&layerCount, availableLayers.data());
	bool layerFound = false;
	std::cout << "Available validation layers: \n";
	for (const char* layerName : validationLayers) {
		for (const auto& layerProperties : availableLayers) {
			std::cout << layerProperties.layerName << std::endl;
			if (strcmp(layerName, layerProperties.layerName) == 0) {
				layerFound = true;
				break;
			}
		}
	}
	if (!layerFound) {
		return false;
	}
	return true;
}

void listExtensions()
{
	// List all extensions
	uint32_t extensionCount = 0;
	vk::enumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

	std::vector<vk::ExtensionProperties> allExtensions(extensionCount);
	vk::enumerateInstanceExtensionProperties(nullptr, &extensionCount, allExtensions.data());

	std::cout << "\navailable extensions:\n";

	for (const auto& extension : allExtensions) {
		std::cout << '\t' << extension.extensionName << '\n';
	}

	// List required extensions
	uint32_t requiredExtensionCount = 0;

	const char** requiredExtensions = glfwGetRequiredInstanceExtensions(&requiredExtensionCount);

	std::cout << "\nrequired extensions:\n";

	for (size_t i = 0; i < requiredExtensionCount; i++) {
		std::cout << "\t\t" << requiredExtensions[i] << std::endl;
	}
}

std::vector<const char*> getRequiredExtensions(const bool enableValidationLayers)
{
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

	if (enableValidationLayers) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	return extensions;
}
