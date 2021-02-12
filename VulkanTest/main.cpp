#define NOMINMAX

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define VULKAN_HPP_NO_NODISCARD_WARNINGS

#define VK_USE_PLATFORM_WIN32_KHR

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

#include "utils.hpp"

const uint32_t WIDTH  = 800;
const uint32_t HEIGHT = 600;

const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
const std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

const vk::PresentModeKHR preferredPresentMode = vk::PresentModeKHR::eMailbox;

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

	vk::SurfaceKHR surface;

	vk::PhysicalDevice physicalDevice = nullptr;
	vk::Device device;

	vk::Queue graphicsQueue;
	vk::Queue presentQueue;

	vk::SwapchainKHR swapChain;
	std::vector<vk::Image> swapChainImages;
	std::vector<vk::ImageView> swapChainImageViews;
	vk::Format swapChainImageFormat;
	vk::Extent2D swapChainExtent;

	vk::RenderPass renderPass;
	vk::PipelineLayout pipelineLayout;
	vk::Pipeline graphicsPipeline;

	std::vector<vk::Framebuffer> swapChainFramebuffers;

	vk::CommandPool commandPool;
	std::vector<vk::CommandBuffer> commandBuffers;

	vk::Semaphore imageAvailableSemaphore;
	vk::Semaphore renderFinishedSemaphore;

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

		createSurface();

		pickPhysicalDevice();
		createLogicalDevice();

		createSwapChain();
		createImageViews();

		createRenderPass();
		createGraphicsPipeline();

		createFramebuffers();
		createCommandPool();
		createCommandBuffers();

		createSemaphores();
	}

	void setupDebugMessenger()
	{
		if (!enableValidationLayers) return;
		vk::DebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		if (CreateDebugUtilsMessengerEXT(&instance, &createInfo, nullptr, &debugMessenger)
			!= vk::Result::eSuccess) {
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}

	void populateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT& createInfo)
	{
		createInfo = {.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
										 | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
										 | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
					  .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
									 | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
									 | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
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

	void createSurface()
	{
		VkSurfaceKHR tempSurface;
		if (glfwCreateWindowSurface(instance, window, nullptr, &tempSurface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
		surface = vk::SurfaceKHR(tempSurface);
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

	int rateDeviceSuitability(const vk::PhysicalDevice device)
	{
		if (!isDeviceSuitable(device)) {
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

	bool isDeviceSuitable(const vk::PhysicalDevice device)
	{
		if (!findQueueFamilies(device).isComplete()) {
			return false;
		}
		if (!checkDeviceExtensionSupport(device)) {
			return false;
		}

		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
		if (swapChainSupport.formats.empty() || swapChainSupport.presentModes.empty()) {
			return false;
		}

		return true;
	}

	bool checkDeviceExtensionSupport(const vk::PhysicalDevice device)
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

	QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice device)
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

	void createLogicalDevice()
	{
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),
												  indices.presentFamily.value()};

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			vk::DeviceQueueCreateInfo queueCreateInfo{
				.queueFamilyIndex = queueFamily, .queueCount = 1, .pQueuePriorities = &queuePriority};
			queueCreateInfos.push_back(queueCreateInfo);
		}

		vk::DeviceQueueCreateInfo queueCreateInfo{.queueFamilyIndex = indices.graphicsFamily.value(),
												  .queueCount       = 1,
												  .pQueuePriorities = &queuePriority};

		vk::PhysicalDeviceFeatures deviceFeatures{};

		vk::DeviceCreateInfo createInfo{
			.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size()),
			.pQueueCreateInfos       = queueCreateInfos.data(),
			.enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size()),
			.ppEnabledExtensionNames = deviceExtensions.data(),
			.pEnabledFeatures        = &deviceFeatures};

		if (enableValidationLayers) {
			createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		} else {
			createInfo.enabledLayerCount = 0;
		}

		if (physicalDevice.createDevice(&createInfo, nullptr, &device) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create logical device!");
		}

		device.getQueue(indices.presentFamily.value(), 0, &presentQueue);
	}

	void createSwapChain()
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

		vk::PresentModeKHR presentMode     = chooseSwapPresentMode(swapChainSupport.presentModes);
		vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		vk::Extent2D extent                = chooseSwapExtent(swapChainSupport.capabilities);

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

		QueueFamilyIndices indices    = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

		if (indices.graphicsFamily != indices.presentFamily) {
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

		createInfo.presentMode = presentMode;
		createInfo.clipped     = VK_TRUE;

		createInfo.oldSwapchain = nullptr;

		if (device.createSwapchainKHR(&createInfo, nullptr, &swapChain) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create swap chain!");
		}

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent      = extent;

		device.getSwapchainImagesKHR(swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		device.getSwapchainImagesKHR(swapChain, &imageCount, swapChainImages.data());
	}

	SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice device)
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

	vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
	{
		// Preferred
		if (std::any_of(availablePresentModes.cbegin(), availablePresentModes.cend(),
						[](auto a) { return (a == preferredPresentMode); })) {
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

	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
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

	void createImageViews()
	{
		swapChainImageViews.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			vk::ImageViewCreateInfo createInfo{
				.image    = swapChainImages[i],
				.viewType = vk::ImageViewType::e2D,
				.format   = swapChainImageFormat,
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

			if (device.createImageView(&createInfo, nullptr, &swapChainImageViews[i])
				!= vk::Result::eSuccess) {
				throw std::runtime_error("failed to create image views!");
			}
		}
	}

	void createGraphicsPipeline()
	{
		auto vertShaderCode = readFile("shaders/vert.spv");
		auto fragShaderCode = readFile("shaders/frag.spv");

		vk::ShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		vk::ShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex,
															  .pName = "main"};

		// WTF
		vertShaderStageInfo.module = vertShaderModule;

		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment,
															  .pName = "main"};
		// Again??
		fragShaderStageInfo.module = fragShaderModule;

		vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{.vertexBindingDescriptionCount   = 0,
															   .pVertexBindingDescriptions      = nullptr,
															   .vertexAttributeDescriptionCount = 0,
															   .pVertexAttributeDescriptions    = nullptr};

		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
			.topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = VK_FALSE};

		vk::Viewport viewport{.x        = 0.0f,
							  .y        = 0.0f,
							  .width    = static_cast<float>(swapChainExtent.width),
							  .height   = static_cast<float>(swapChainExtent.height),
							  .minDepth = 0.0f,
							  .maxDepth = 1.0f};

		vk::Rect2D scissor{.offset = {0, 0}, .extent = swapChainExtent};

		vk::PipelineViewportStateCreateInfo viewportState{
			.viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor};

		vk::PipelineRasterizationStateCreateInfo rasterizer{.depthClampEnable        = VK_FALSE,
															.rasterizerDiscardEnable = VK_FALSE,
															.polygonMode             = vk::PolygonMode::eFill,
															.cullMode        = vk::CullModeFlagBits::eBack,
															.frontFace       = vk::FrontFace::eClockwise,
															.depthBiasEnable = VK_FALSE,
															.depthBiasConstantFactor = 0.0f,
															.depthBiasClamp          = 0.0f,
															.depthBiasSlopeFactor    = 0.0f,
															.lineWidth               = 1.0f};

		vk::PipelineMultisampleStateCreateInfo multisampling{
			.rasterizationSamples  = vk::SampleCountFlagBits::e1,
			.sampleShadingEnable   = VK_FALSE,
			.minSampleShading      = 1.0f,
			.pSampleMask           = nullptr,
			.alphaToCoverageEnable = VK_FALSE,
			.alphaToOneEnable      = VK_FALSE};

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

		vk::PipelineColorBlendStateCreateInfo colorBlending{.logicOpEnable   = VK_FALSE,
															.logicOp         = vk::LogicOp::eCopy,
															.attachmentCount = 1,
															.pAttachments    = &colorBlendAttachment};
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		vk::DynamicState dynamicStates[] = {vk::DynamicState::eViewport, vk::DynamicState::eLineWidth};

		vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = 2,
														.pDynamicStates    = dynamicStates};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount         = 0,
														.pSetLayouts            = nullptr,
														.pushConstantRangeCount = 0,
														.pPushConstantRanges    = nullptr};

		if (device.createPipelineLayout(&pipelineLayoutInfo, nullptr, &pipelineLayout)
			!= vk::Result::eSuccess) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		vk::GraphicsPipelineCreateInfo pipelineInfo{.stageCount          = 2,
													.pStages             = shaderStages,
													.pVertexInputState   = &vertexInputInfo,
													.pInputAssemblyState = &inputAssembly,
													.pViewportState      = &viewportState,
													.pRasterizationState = &rasterizer,
													.pMultisampleState   = &multisampling,
													.pDepthStencilState  = nullptr,
													.pColorBlendState    = &colorBlending,
													.pDynamicState       = nullptr,
													.layout              = pipelineLayout,
													.renderPass          = renderPass,
													.subpass             = 0,
													.basePipelineHandle  = nullptr,
													.basePipelineIndex   = -1};

		if (device.createGraphicsPipelines(nullptr, 1, &pipelineInfo, nullptr, &graphicsPipeline)
			!= vk::Result::eSuccess) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		device.destroyShaderModule(fragShaderModule, nullptr);
		device.destroyShaderModule(vertShaderModule, nullptr);
	}

	vk::ShaderModule createShaderModule(const std::vector<char>& code)
	{
		vk::ShaderModuleCreateInfo createInfo{.codeSize = code.size(),
											  .pCode    = reinterpret_cast<const uint32_t*>(code.data())};
		vk::ShaderModule shaderModule;
		if (device.createShaderModule(&createInfo, nullptr, &shaderModule) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create shader module!");
		}
		return shaderModule;
	}

	void createRenderPass()
	{
		vk::AttachmentDescription colorAttachment{.format         = swapChainImageFormat,
												  .samples        = vk::SampleCountFlagBits::e1,
												  .loadOp         = vk::AttachmentLoadOp::eClear,
												  .storeOp        = vk::AttachmentStoreOp::eStore,
												  .stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
												  .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
												  .initialLayout  = vk::ImageLayout::eUndefined,
												  .finalLayout    = vk::ImageLayout::ePresentSrcKHR};

		vk::AttachmentReference colorAttachmentRef{.attachment = 0,
												   .layout     = vk::ImageLayout::eColorAttachmentOptimal};

		vk::SubpassDescription subpass{.pipelineBindPoint    = vk::PipelineBindPoint::eGraphics,
									   .colorAttachmentCount = 1,
									   .pColorAttachments    = &colorAttachmentRef};

		vk::SubpassDependency dependency{.srcSubpass    = VK_SUBPASS_EXTERNAL,
										 .dstSubpass    = 0,
										 .srcStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput,
										 .dstStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput,
										 .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite};

		vk::RenderPassCreateInfo renderPassInfo{.attachmentCount = 1,
												.pAttachments    = &colorAttachment,
												.subpassCount    = 1,
												.pSubpasses      = &subpass,
												.dependencyCount = 1,
												.pDependencies   = &dependency};

		if (device.createRenderPass(&renderPassInfo, nullptr, &renderPass) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create render pass!");
		}
	}

	void createFramebuffers()
	{
		swapChainFramebuffers.resize(swapChainImageViews.size());
		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			vk::ImageView attachments[] = {swapChainImageViews[i]};

			vk::FramebufferCreateInfo framebufferInfo{.renderPass      = renderPass,
													  .attachmentCount = 1,
													  .pAttachments    = attachments,
													  .width           = swapChainExtent.width,
													  .height          = swapChainExtent.height,
													  .layers          = 1};
			if (device.createFramebuffer(&framebufferInfo, nullptr, &swapChainFramebuffers[i])
				!= vk::Result::eSuccess) {
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}

	void createCommandPool()
	{
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		vk::CommandPoolCreateInfo poolInfo{.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()};

		if (device.createCommandPool(&poolInfo, nullptr, &commandPool) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create command pool!");
		}
	}

	void createCommandBuffers()
	{
		commandBuffers.resize(swapChainFramebuffers.size());
		vk::CommandBufferAllocateInfo allocInfo{.commandPool        = commandPool,
												.level              = vk::CommandBufferLevel::ePrimary,
												.commandBufferCount = (uint32_t)commandBuffers.size()};

		if (device.allocateCommandBuffers(&allocInfo, commandBuffers.data()) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to allocate command buffers!");
		}

		for (size_t i = 0; i < commandBuffers.size(); i++) {
			vk::CommandBufferBeginInfo beginInfo{};
			if (commandBuffers[i].begin(&beginInfo) != vk::Result::eSuccess) {
				throw std::runtime_error("failed to begin recording command buffer!");
			}

			vk::ClearValue clearColor = {std::array<float, 4>({{0.0f, 0.0f, 0.0f, 1.0f}})};

			vk::RenderPassBeginInfo renderPassInfo{.renderPass      = renderPass,
												   .framebuffer     = swapChainFramebuffers[i],
												   .clearValueCount = 1,
												   .pClearValues    = &clearColor};

			renderPassInfo.renderArea.offset = VkOffset2D{0, 0};
			renderPassInfo.renderArea.extent = swapChainExtent;

			commandBuffers[i].beginRenderPass(&renderPassInfo, vk::SubpassContents::eInline);

			commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
			commandBuffers[i].draw(3, 1, 0, 0);

			commandBuffers[i].endRenderPass();

			// Presumably has built in fail check
			commandBuffers[i].end();
		}
	}

	void createSemaphores()
	{
		vk::SemaphoreCreateInfo semaphoreInfo{};
		if (device.createSemaphore(&semaphoreInfo, nullptr, &imageAvailableSemaphore) != vk::Result::eSuccess
			|| device.createSemaphore(&semaphoreInfo, nullptr, &renderFinishedSemaphore)
				   != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create semaphores!");
		}
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
			drawFrame();
		}

		device.waitIdle();
	}

	void drawFrame()
	{
		uint32_t imageIndex;
		device.acquireNextImageKHR(swapChain, UINT64_MAX, imageAvailableSemaphore, nullptr, &imageIndex);

		vk::Semaphore waitSemaphores[]      = {imageAvailableSemaphore};
		vk::Semaphore signalSemaphores[]    = {renderFinishedSemaphore};
		vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};

		vk::SubmitInfo submitInfo{.waitSemaphoreCount   = 1,
								  .pWaitSemaphores      = waitSemaphores,
								  .pWaitDstStageMask    = waitStages,
								  .commandBufferCount   = 1,
								  .pCommandBuffers      = &commandBuffers[imageIndex],
								  .signalSemaphoreCount = 1,
								  .pSignalSemaphores    = signalSemaphores};

		if (graphicsQueue.submit(1, &submitInfo, nullptr) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		vk::SwapchainKHR swapChains[] = {swapChain};

		vk::PresentInfoKHR presentInfo{.waitSemaphoreCount = 1,
									   .pWaitSemaphores    = signalSemaphores,
									   .swapchainCount     = 1,
									   .pSwapchains        = swapChains,
									   .pImageIndices      = &imageIndex,
									   .pResults           = nullptr};

		presentQueue.presentKHR(&presentInfo);
	}

	void cleanup()
	{
		std::cout << std::endl;

		device.destroySemaphore(renderFinishedSemaphore, nullptr);
		device.destroySemaphore(imageAvailableSemaphore, nullptr);

		device.destroyCommandPool(commandPool, nullptr);

		for (auto framebuffer : swapChainFramebuffers) {
			device.destroyFramebuffer(framebuffer, nullptr);
		}

		device.destroyPipeline(graphicsPipeline, nullptr);
		device.destroyPipelineLayout(pipelineLayout, nullptr);
		device.destroyRenderPass(renderPass, nullptr);

		for (auto imageView : swapChainImageViews) {
			device.destroyImageView(imageView, nullptr);
		}

		device.destroySwapchainKHR(swapChain, nullptr);

		device.destroy();

		instance.destroySurfaceKHR(surface, nullptr);

		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(&instance, debugMessenger, nullptr);
		}

		instance.destroy();

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
