#include "../Defines.h"

#include "VulkanInit.hpp"

#include <GLFW/glfw3.h>

#include <iostream>

bool checkValidationLayerSupport(const std::vector<const char*>& validationLayers)
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

std::vector<const char*> getRequiredExtensions()
{
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

#ifdef ENABLE_VALIDATION_LAYERS
	extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

	return extensions;
}

static VKAPI_ATTR vk::Bool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
			  VkDebugUtilsMessageTypeFlagsEXT messageType,
			  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
			  void* pUserData)
{
	std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

	return VK_FALSE;
}

vk::DebugUtilsMessengerCreateInfoEXT populateDebugMessengerCreateInfo()
{
	return {.messageSeverity = /*vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
									 | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
									 | */
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
			.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
						   | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
						   | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
			.pfnUserCallback = debugCallback};
}

vk::Instance createInstance(const std::vector<const char*>& validationLayers)
{
	vk::Instance instance;
#ifdef ENABLE_VALIDATION_LAYERS
	if (!checkValidationLayerSupport(validationLayers)) {
		throw std::runtime_error("validation layers requested, but not available!");
	}
#endif
	// Create App info
	vk::ApplicationInfo appInfo;
	appInfo.pApplicationName   = "Hello Triangle";
	appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
	appInfo.pEngineName        = "Engineer";
	appInfo.engineVersion      = VK_MAKE_VERSION(0, 1, 0);
	appInfo.apiVersion         = VK_API_VERSION_1_2;

	vk::InstanceCreateInfo createInfo;
	createInfo.pApplicationInfo = &appInfo;

	std::vector<const char*> extensions = getRequiredExtensions();
	createInfo.enabledExtensionCount    = static_cast<uint32_t>(extensions.size());
	createInfo.ppEnabledExtensionNames  = extensions.data();

	vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo = populateDebugMessengerCreateInfo();

#ifdef ENABLE_VALIDATION_LAYERS
	createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
	createInfo.ppEnabledLayerNames = validationLayers.data();
	createInfo.pNext               = static_cast<vk::DebugUtilsMessengerCreateInfoEXT*>(&debugCreateInfo);
#else
	createInfo.enabledLayerCount = 0;
	createInfo.pNext             = nullptr;
#endif
	if (vk::createInstance(&createInfo, nullptr, &instance) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create instance!");
	}

	listExtensions();

	return instance;
}

vk::Result CreateDebugUtilsMessengerEXT(vk::Instance& instance,
										const vk::DebugUtilsMessengerCreateInfoEXT* pCreateInfo,
										const vk::AllocationCallbacks* pAllocator,
										vk::DebugUtilsMessengerEXT* pDebugMessenger)
{
	auto dldi = vk::DispatchLoaderDynamic(instance, vkGetInstanceProcAddr);

	return instance.createDebugUtilsMessengerEXT(pCreateInfo, pAllocator, pDebugMessenger, dldi);
}

void DestroyDebugUtilsMessengerEXT(vk::Instance& instance,
								   vk::DebugUtilsMessengerEXT& debugMessenger,
								   const vk::AllocationCallbacks* pAllocator)
{
	auto dldi = vk::DispatchLoaderDynamic(instance, vkGetInstanceProcAddr);

	instance.destroyDebugUtilsMessengerEXT(debugMessenger, pAllocator, dldi);
}

vk::DebugUtilsMessengerEXT setupDebugMessenger(vk::Instance& instance)
{
	vk::DebugUtilsMessengerEXT debugMessenger;
#ifndef ENABLE_VALIDATION_LAYERS
	return {};
#endif
	vk::DebugUtilsMessengerCreateInfoEXT createInfo = populateDebugMessengerCreateInfo();

	if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger)
		!= vk::Result::eSuccess) {
		throw std::runtime_error("failed to set up debug messenger!");
	}

	return debugMessenger;
}

vk::SurfaceKHR createSurface(vk::Instance& instance, GLFWwindow* window)
{
	VkSurfaceKHR tempSurface;
	if (glfwCreateWindowSurface(instance, window, nullptr, &tempSurface) != VK_SUCCESS) {
		throw std::runtime_error("failed to create window surface!");
	}
	return vk::SurfaceKHR(tempSurface);
}
