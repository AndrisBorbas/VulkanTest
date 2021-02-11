#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define VULKAN_HPP_NO_NODISCARD_WARNINGS

#define VK_USE_PLATFORM_WIN32_KHR

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <vulkan/vulkan.hpp>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

const uint32_t WIDTH  = 800;
const uint32_t HEIGHT = 600;

const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

vk::Result CreateDebugUtilsMessengerEXT(vk::Instance* instance,
										const vk::DebugUtilsMessengerCreateInfoEXT* pCreateInfo,
										const vk::AllocationCallbacks* pAllocator,
										vk::DebugUtilsMessengerEXT* pDebugMessenger)
{
	auto dldi = vk::DispatchLoaderDynamic(*instance, vkGetInstanceProcAddr);

	return instance->createDebugUtilsMessengerEXT(pCreateInfo, pAllocator, pDebugMessenger, dldi);
}

void DestroyDebugUtilsMessengerEXT(vk::Instance* instance,
								   vk::DebugUtilsMessengerEXT debugMessenger,
								   const vk::AllocationCallbacks* pAllocator)
{
	auto dldi = vk::DispatchLoaderDynamic(*instance, vkGetInstanceProcAddr);

	instance->destroyDebugUtilsMessengerEXT(debugMessenger, pAllocator, dldi);
}

class HelloTriangleApplication
{
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	GLFWwindow* window;
	vk::Instance instance;
	vk::DebugUtilsMessengerEXT debugMessenger;
	vk::PhysicalDevice physicalDevice = nullptr;
	vk::Device device;
	vk::Queue graphicsQueue;
	struct QueueFamilyIndices
	{
		std::optional<uint32_t> graphicsFamily;
		bool isComplete() { return graphicsFamily.has_value(); }
	};

	vk::SurfaceKHR surface;

	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	}
	void initVulkan()
	{
		createInstance();
		setupDebugMessenger();
		pickPhysicalDevice();
		createLogicalDevice();
	}

	void setupDebugMessenger()
	{
		if (!enableValidationLayers) return;
		vk::DebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		if (CreateDebugUtilsMessengerEXT(&instance, &createInfo, nullptr, &debugMessenger) !=
			vk::Result::eSuccess) {
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}

	void populateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT& createInfo)
	{
		createInfo = {.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
										 vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
										 vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
					  .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
									 vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
									 vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
					  .pfnUserCallback = debugCallback};
	}

	void createInstance()
	{
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested, but not available!");
		}

		// Create App info
		vk::ApplicationInfo appInfo{.pApplicationName   = "Hello Triangle",
									.applicationVersion = VK_MAKE_VERSION(0, 1, 0),
									.pEngineName        = "Engineer",
									.engineVersion      = VK_MAKE_VERSION(0, 1, 0),
									.apiVersion         = VK_API_VERSION_1_2};

		vk::InstanceCreateInfo createInfo{.pApplicationInfo = &appInfo};

		// Set required extensions
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;

		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount    = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames  = extensions.data();

		vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo;
		if (enableValidationLayers) {
			createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();

			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = static_cast<vk::DebugUtilsMessengerCreateInfoEXT*>(&debugCreateInfo);
		} else {
			createInfo.enabledLayerCount = 0;
			createInfo.pNext             = nullptr;
		}

		if (vk::createInstance(&createInfo, nullptr, &instance) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create instance!");
		}

		listExtensions();
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

	bool checkValidationLayerSupport()
	{
		uint32_t layerCount;
		vk::enumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<vk::LayerProperties> availableLayers(layerCount);
		vk::enumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers) {
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}
			if (!layerFound) {
				return false;
			}
		}
		return true;
	}

	void pickPhysicalDevice()
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

		for (const auto& device : devices) {
			int score = rateDeviceSuitability(device);
			candidates.insert(std::make_pair(score, device));
		}

		// Check if the best candidate is suitable at all
		if (candidates.rbegin()->first > 0) {
			physicalDevice = candidates.rbegin()->second;
			std::cout << physicalDevice.getProperties().deviceName << std::endl;
		} else {
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	int rateDeviceSuitability(vk::PhysicalDevice device)
	{
		if (!findQueueFamilies(device).isComplete()) {
			return 0;
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

	QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device)
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
			// early exit
			if (indices.isComplete()) {
				break;
			}

			i++;
		}

		return indices;
	}

	void createLogicalDevice()
	{
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		float queuePriority = 1.0f;

		vk::DeviceQueueCreateInfo queueCreateInfo{.queueFamilyIndex = indices.graphicsFamily.value(),
												  .queueCount       = 1,
												  .pQueuePriorities = &queuePriority};

		vk::PhysicalDeviceFeatures deviceFeatures{};

		vk::DeviceCreateInfo createInfo{.queueCreateInfoCount  = 1,
										.pQueueCreateInfos     = &queueCreateInfo,
										.enabledExtensionCount = 0,
										.pEnabledFeatures      = &deviceFeatures};

		if (enableValidationLayers) {
			createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		} else {
			createInfo.enabledLayerCount = 0;
		}

		if (physicalDevice.createDevice(&createInfo, nullptr, &device) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create logical device!");
		}

		device.getQueue(indices.graphicsFamily.value(), 0, &graphicsQueue);
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

	std::vector<const char*> getRequiredExtensions()
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

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	void cleanup()
	{
		std::cout << std::endl;

		vkDestroyDevice(device, nullptr);

		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(&instance, debugMessenger, nullptr);
		}

		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();
	}
};

int main()
{
	HelloTriangleApplication app;

	try {
		app.run();
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
