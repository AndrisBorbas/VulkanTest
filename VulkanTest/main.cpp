#include "Defines.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <stb_image.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include "Shaders.hpp"
#include "Vulkan/VulkanDevices.hpp"
#include "Vulkan/VulkanInit.hpp"
#include "Vulkan/VulkanRendering.hpp"
#include "utils.hpp"

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

const uint32_t WIDTH  = 800;
const uint32_t HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
const std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

const vk::PresentModeKHR preferredPresentMode = vk::PresentModeKHR::eImmediate;

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
	GLFWwindow* window_;

	ImGui_ImplVulkanH_Window MainWindowData_;
	bool show_demo_window    = true;
	bool show_another_window = false;
	ImVec4 clear_color       = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	vk::Instance instance_;

	vk::DebugUtilsMessengerEXT debugMessenger_;

	vk::SurfaceKHR surface_;

	vk::PhysicalDevice physicalDevice_;
	vk::Device device_;

	vk::Queue graphicsQueue_;
	vk::Queue presentQueue_;

	vk::SwapchainKHR swapchain_;
	std::vector<vk::Image> swapchainImages_;
	std::vector<vk::ImageView> swapchainImageViews_;
	vk::Format swapchainImageFormat_;
	vk::Extent2D swapchainExtent_;

	vk::RenderPass renderPass_;
	vk::DescriptorSetLayout descriptorSetLayout_;
	vk::DescriptorPool descriptorPool_;
	std::vector<vk::DescriptorSet> descriptorSets_;
	vk::PipelineLayout pipelineLayout_;
	vk::Pipeline graphicsPipeline_;

	vk::SampleCountFlagBits msaaSamples_ = vk::SampleCountFlagBits::e1;

	std::vector<vk::Framebuffer> swapChainFramebuffers_;

	vk::CommandPool commandPool_;
	std::vector<vk::CommandBuffer> commandBuffers_;

	std::vector<vk::Semaphore> imageAvailableSemaphores_;
	std::vector<vk::Semaphore> renderFinishedSemaphores_;
	std::vector<vk::Fence> inFlightFences_;
	std::vector<vk::Fence> imagesInFlight_;

	vk::Image colorImage_;
	vk::DeviceMemory colorImageMemory_;
	vk::ImageView colorImageView_;

	std::vector<Vertex> vertices_;
	std::vector<uint32_t> indices_;

	vk::Buffer vertexBuffer_;
	vk::DeviceMemory vertexBufferMemory_;
	vk::Buffer indexBuffer_;
	vk::DeviceMemory indexBufferMemory_;

	std::vector<vk::Buffer> uniformBuffers_;
	std::vector<vk::DeviceMemory> uniformBuffersMemory_;

	uint32_t mipLevels_;
	vk::Image textureImage_;
	vk::DeviceMemory textureImageMemory_;

	vk::ImageView textureImageView_;
	vk::Sampler textureSampler_;

	vk::Image depthImage_;
	vk::DeviceMemory depthImageMemory_;
	vk::ImageView depthImageView_;

	size_t currentFrame_ = 0;

	bool framebufferResized_ = false;

	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window_ = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window_, this);
		glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
	{
		auto* app                = static_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
		app->framebufferResized_ = true;
	}

	void initImgui()
	{
		std::array<vk::DescriptorPoolSize, 11> poolSizes = {{
			{vk::DescriptorType::eSampler, 1000},
			{vk::DescriptorType::eCombinedImageSampler, 1000},
			{vk::DescriptorType::eSampledImage, 1000},
			{vk::DescriptorType::eStorageImage, 1000},
			{vk::DescriptorType::eUniformTexelBuffer, 1000},
			{vk::DescriptorType::eStorageTexelBuffer, 1000},
			{vk::DescriptorType::eUniformBuffer, 1000},
			{vk::DescriptorType::eStorageBuffer, 1000},
			{vk::DescriptorType::eUniformBufferDynamic, 1000},
			{vk::DescriptorType::eStorageBufferDynamic, 1000},
			{vk::DescriptorType::eInputAttachment, 1000},
		}};

		vk::DescriptorPoolCreateInfo pool_info = {
			.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
			.maxSets       = 1000,
			.poolSizeCount = poolSizes.size(),
			.pPoolSizes    = poolSizes.data(),
		};

		vk::DescriptorPool imguiPool;
		if (device_.createDescriptorPool(&pool_info, nullptr, &imguiPool) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create imgui descriptor pool!");
		}

		// 2: initialize imgui library

		// this initializes the core structures of imgui
		ImGui::CreateContext();

		ImGui_ImplGlfw_InitForVulkan(window_, true);

		// this initializes imgui for Vulkan
		ImGui_ImplVulkan_InitInfo init_info = {
			.Instance       = instance_,
			.PhysicalDevice = physicalDevice_,
			.Device         = device_,
			//.QueueFamily     = g_QueueFamily,
			.Queue = graphicsQueue_,
			//.PipelineCache   = g_PipelineCache,
			.DescriptorPool = imguiPool,
			.MinImageCount  = 3,
			.ImageCount     = 3,
			.Allocator      = nullptr,
			//.CheckVkResultFn = check_vk_result,
		};

		ImGui_ImplVulkan_Init(&init_info, renderPass_);

		// execute a gpu command to upload imgui font textures
		// immediate_submit([&](VkCommandBuffer cmd) { ImGui_ImplVulkan_CreateFontsTexture(cmd); });

		vk::CommandBuffer commandBuffer = beginSingleTimeCommands(device_, commandPool_);
		ImGui_ImplVulkan_CreateFontsTexture(commandBuffer);
		endSingleTimeCommands(device_, commandBuffer, graphicsQueue_, commandPool_);

		// clear font textures from cpu data
		ImGui_ImplVulkan_DestroyFontUploadObjects();

		// add the destroy the imgui created structures

		// device_.destroyDescriptorPool( imguiPool, nullptr);
		// ImGui_ImplVulkan_Shutdown();
	}

	void initVulkan()
	{
		instance_       = createInstance(validationLayers);
		debugMessenger_ = setupDebugMessenger(instance_);

		surface_ = createSurface(instance_, window_);

		physicalDevice_ = pickPhysicalDevice(instance_, surface_, deviceExtensions, msaaSamples_);
		device_ = createLogicalDevice(instance_, surface_, physicalDevice_, graphicsQueue_, presentQueue_,
									  deviceExtensions, validationLayers);

		swapchain_           = createSwapChain(device_, physicalDevice_, window_, surface_, swapchainImages_,
                                     swapchainImageFormat_, swapchainExtent_, preferredPresentMode);
		swapchainImageViews_ = createImageViews(device_, swapchainImages_, swapchainImageFormat_, 1);
		renderPass_ = createRenderPass(device_, physicalDevice_, swapchainImageFormat_, msaaSamples_);
		descriptorSetLayout_ = createDescriptorSetLayout(device_);
		graphicsPipeline_    = createGraphicsPipeline(device_, swapchainExtent_, descriptorSetLayout_,
                                                   pipelineLayout_, renderPass_, msaaSamples_);

		commandPool_ = createCommandPool(device_, physicalDevice_, surface_);
		loadModel("assets/models/viking_room.obj");
		std::tie(colorImage_, colorImageMemory_, colorImageView_) = createColorResources(
			device_, physicalDevice_, swapchainImageFormat_, swapchainExtent_, msaaSamples_);
		std::tie(depthImage_, depthImageMemory_, depthImageView_) = createDepthResources(
			device_, physicalDevice_, swapchainExtent_, graphicsQueue_, commandPool_, msaaSamples_);
		createFramebuffers(device_, swapChainFramebuffers_, swapchainImageViews_, colorImageView_,
						   depthImageView_, renderPass_, swapchainExtent_);
		std::tie(textureImage_, textureImageMemory_) =
			createTextureImage(device_, physicalDevice_, "assets/textures/viking_room.png", STBI_rgb_alpha,
							   mipLevels_, vk::Format::eR8G8B8A8Srgb, graphicsQueue_, commandPool_);
		textureImageView_ = createImageView(device_, textureImage_, vk::Format::eR8G8B8A8Srgb,
											vk::ImageAspectFlagBits::eColor, mipLevels_);
		textureSampler_   = createTextureSampler(device_, physicalDevice_, mipLevels_);

		createVertexBuffer(device_, physicalDevice_, commandPool_, graphicsQueue_, vertices_, vertexBuffer_,
						   vertexBufferMemory_);
		createIndexBuffer(device_, physicalDevice_, commandPool_, graphicsQueue_, indices_, indexBuffer_,
						  indexBufferMemory_);
		createUniformBuffers(device_, physicalDevice_, uniformBuffers_, uniformBuffersMemory_,
							 swapchainImages_);
		descriptorPool_ = createDescriptorPool(device_, swapchainImages_);
		createDescriptorSets(device_, descriptorSetLayout_, descriptorSets_, swapchainImages_,
							 descriptorPool_, uniformBuffers_, textureImageView_, textureSampler_);
		createCommandBuffers(device_, commandBuffers_, swapChainFramebuffers_, commandPool_, renderPass_,
							 swapchainExtent_, graphicsPipeline_, vertexBuffer_, indexBuffer_,
							 pipelineLayout_, descriptorSets_, indices_);

		createSyncObjects(device_, imageAvailableSemaphores_, renderFinishedSemaphores_, inFlightFences_,
						  imagesInFlight_, MAX_FRAMES_IN_FLIGHT, swapchainImages_);
		initImgui();
	}

	void cleanupSwapChain()
	{
		device_.destroyImageView(colorImageView_, nullptr);
		device_.destroyImage(colorImage_, nullptr);
		device_.freeMemory(colorImageMemory_, nullptr);

		device_.destroyImageView(depthImageView_, nullptr);
		device_.destroyImage(depthImage_, nullptr);
		device_.freeMemory(depthImageMemory_, nullptr);

		for (auto& swapChainFramebuffer : swapChainFramebuffers_) {
			device_.destroyFramebuffer(swapChainFramebuffer, nullptr);
		}

		device_.freeCommandBuffers(commandPool_, static_cast<uint32_t>(commandBuffers_.size()),
								   commandBuffers_.data());

		device_.destroyPipeline(graphicsPipeline_, nullptr);
		device_.destroyPipelineLayout(pipelineLayout_, nullptr);
		device_.destroyRenderPass(renderPass_, nullptr);

		for (auto& swapChainImageView : swapchainImageViews_) {
			device_.destroyImageView(swapChainImageView, nullptr);
		}

		device_.destroySwapchainKHR(swapchain_, nullptr);

		for (size_t i = 0; i < swapchainImages_.size(); i++) {
			device_.destroyBuffer(uniformBuffers_[i], nullptr);
			device_.freeMemory(uniformBuffersMemory_[i], nullptr);
		}

		device_.destroyDescriptorPool(descriptorPool_, nullptr);
	}

	void recreateSwapChain()
	{
		int width = 0, height = 0;
		glfwGetFramebufferSize(window_, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window_, &width, &height);
			glfwWaitEvents();
		}

		ImGui_ImplVulkan_SetMinImageCount(3);
		ImGui_ImplVulkanH_CreateOrResizeWindow(instance_, physicalDevice_, device_, &MainWindowData_,
											   (uint32_t)-1, nullptr, width, height, 3);

		vkDeviceWaitIdle(device_);

		cleanupSwapChain();

		swapchain_           = createSwapChain(device_, physicalDevice_, window_, surface_, swapchainImages_,
                                     swapchainImageFormat_, swapchainExtent_, preferredPresentMode);
		swapchainImageViews_ = createImageViews(device_, swapchainImages_, swapchainImageFormat_, 1);
		renderPass_       = createRenderPass(device_, physicalDevice_, swapchainImageFormat_, msaaSamples_);
		graphicsPipeline_ = createGraphicsPipeline(device_, swapchainExtent_, descriptorSetLayout_,
												   pipelineLayout_, renderPass_, msaaSamples_);
		std::tie(colorImage_, colorImageMemory_, colorImageView_) = createColorResources(
			device_, physicalDevice_, swapchainImageFormat_, swapchainExtent_, msaaSamples_);
		std::tie(depthImage_, depthImageMemory_, depthImageView_) = createDepthResources(
			device_, physicalDevice_, swapchainExtent_, graphicsQueue_, commandPool_, msaaSamples_);
		createFramebuffers(device_, swapChainFramebuffers_, swapchainImageViews_, colorImageView_,
						   depthImageView_, renderPass_, swapchainExtent_);
		createUniformBuffers(device_, physicalDevice_, uniformBuffers_, uniformBuffersMemory_,
							 swapchainImages_);
		descriptorPool_ = createDescriptorPool(device_, swapchainImages_);
		createDescriptorSets(device_, descriptorSetLayout_, descriptorSets_, swapchainImages_,
							 descriptorPool_, uniformBuffers_, textureImageView_, textureSampler_);
		createCommandBuffers(device_, commandBuffers_, swapChainFramebuffers_, commandPool_, renderPass_,
							 swapchainExtent_, graphicsPipeline_, vertexBuffer_, indexBuffer_,
							 pipelineLayout_, descriptorSets_, indices_);
	}

	void loadModel(const char* filename)
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn, err;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename)) {
			throw std::runtime_error(warn + err);
		}

		std::unordered_map<Vertex, uint32_t> uniqueVertices{};

		for (const auto& shape : shapes) {
			for (const auto& index : shape.mesh.indices) {
				Vertex vertex{};

				vertex.pos = {
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2],
				};

				vertex.texCoord = {
					attrib.texcoords[2 * index.texcoord_index + 0],
					1.0f - attrib.texcoords[2 * index.texcoord_index + 1],
				};

				vertex.color = {1.0f, 1.0f, 1.0f};

				if (uniqueVertices.count(vertex) == 0) {
					uniqueVertices[vertex] = static_cast<uint32_t>(vertices_.size());
					vertices_.push_back(vertex);
				}
				indices_.push_back(uniqueVertices[vertex]);
			}
		}
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window_)) {
			glfwPollEvents();
			drawFrame();
		}

		device_.waitIdle();
	}

	void drawFrame()
	{
		device_.waitForFences(1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		vk::Result result = device_.acquireNextImageKHR(
			swapchain_, UINT64_MAX, imageAvailableSemaphores_[currentFrame_], nullptr, &imageIndex);

		if (result == vk::Result::eErrorOutOfDateKHR) {
			recreateSwapChain();
			return;
		} else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		// Check if a previous frame is using this image (i.e. there is its fence to wait on)
		if (imagesInFlight_[imageIndex] != VK_NULL_HANDLE) {
			device_.waitForFences(1, &imagesInFlight_[imageIndex], VK_TRUE, UINT64_MAX);
		}
		// Mark the image as now being in use by this frame
		imagesInFlight_[imageIndex] = inFlightFences_[currentFrame_];

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		if (show_demo_window) ImGui::ShowDemoWindow(&show_demo_window);

		{
			static float f     = 0.0f;
			static int counter = 0;

			ImGui::Begin("Hello, world!");  // Create a window called "Hello, world!" and append into it.

			ImGui::Text("This is some useful text.");  // Display some text (you can use a format strings too)
			ImGui::Checkbox("Demo Window",
							&show_demo_window);  // Edit bools storing our window open/close state
			ImGui::Checkbox("Another Window", &show_another_window);

			ImGui::SliderFloat("float", &f, 0.0f, 1.0f);  // Edit 1 float using a slider from 0.0f to 1.0f
			ImGui::ColorEdit3("clear color", (float*)&clear_color);  // Edit 3 floats representing a color

			if (ImGui::Button("Button"))  // Buttons return true when clicked (most widgets return true when
										  // edited/activated)
				counter++;
			ImGui::SameLine();
			ImGui::Text("counter = %d", counter);

			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
						ImGui::GetIO().Framerate);
			ImGui::End();
		}
		if (show_another_window) {
			ImGui::Begin("Another Window",
						 &show_another_window);  // Pass a pointer to our bool variable (the window will have
												 // a closing button that will clear the bool when clicked)
			ImGui::Text("Hello from another window!");
			if (ImGui::Button("Close Me")) show_another_window = false;
			ImGui::End();
		}

		ImGui::Render();

		for (size_t i = 0; i < commandBuffers_.size(); i++) {
			vk::CommandBufferBeginInfo beginInfo{};
			if (commandBuffers_[i].begin(&beginInfo) != vk::Result::eSuccess) {
				throw std::runtime_error("failed to begin recording command buffer!");
			}

			std::array<vk::ClearValue, 2> clearValues{};
			clearValues[0].setColor(std::array<float, 4>{0.01f, 0.01f, 0.0125f, 0.0f});
			clearValues[1].setDepthStencil({1.0f, 0});

			vk::RenderPassBeginInfo renderPassInfo{
				.renderPass      = renderPass_,
				.framebuffer     = swapChainFramebuffers_[i],
				.clearValueCount = static_cast<uint32_t>(clearValues.size()),
				.pClearValues    = clearValues.data()};

			renderPassInfo.renderArea.offset = VkOffset2D{0, 0};
			renderPassInfo.renderArea.extent = swapchainExtent_;

			commandBuffers_[i].beginRenderPass(&renderPassInfo, vk::SubpassContents::eInline);

			commandBuffers_[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline_);

			vk::Buffer vertexBuffers[] = {vertexBuffer_};
			vk::DeviceSize offsets[]   = {0};
			commandBuffers_[i].bindVertexBuffers(0, 1, vertexBuffers, offsets);
			commandBuffers_[i].bindIndexBuffer(indexBuffer_, 0, vk::IndexType::eUint32);

			commandBuffers_[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout_, 0, 1,
												  &descriptorSets_[i], 0, nullptr);
			commandBuffers_[i].drawIndexed(static_cast<uint32_t>(indices_.size()), 1, 0, 0, 0);

			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffers_[i]);

			commandBuffers_[i].endRenderPass();

			// Presumably has built in fail check
			commandBuffers_[i].end();
		}

		vk::Semaphore waitSemaphores[]      = {imageAvailableSemaphores_[currentFrame_]};
		vk::Semaphore signalSemaphores[]    = {renderFinishedSemaphores_[currentFrame_]};
		vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};

		updateUniformBuffer(imageIndex, device_, uniformBuffersMemory_, swapchainExtent_);

		vk::SubmitInfo submitInfo{.waitSemaphoreCount   = 1,
								  .pWaitSemaphores      = waitSemaphores,
								  .pWaitDstStageMask    = waitStages,
								  .commandBufferCount   = 1,
								  .pCommandBuffers      = &commandBuffers_[imageIndex],
								  .signalSemaphoreCount = 1,
								  .pSignalSemaphores    = signalSemaphores};

		device_.resetFences(1, &inFlightFences_[currentFrame_]);

		if (graphicsQueue_.submit(1, &submitInfo, inFlightFences_[currentFrame_]) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		vk::SwapchainKHR swapChains[] = {swapchain_};

		vk::PresentInfoKHR presentInfo{.waitSemaphoreCount = 1,
									   .pWaitSemaphores    = signalSemaphores,
									   .swapchainCount     = 1,
									   .pSwapchains        = swapChains,
									   .pImageIndices      = &imageIndex,
									   .pResults           = nullptr};

		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffers_[imageIndex]);

		result = presentQueue_.presentKHR(&presentInfo);

		if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR
			|| framebufferResized_) {
			framebufferResized_ = false;
			recreateSwapChain();
		} else if (result != vk::Result::eSuccess) {
			throw std::runtime_error("failed to present swap chain image!");
		}

		currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void cleanup()
	{
		std::cout << std::endl;
		cleanupSwapChain();

		device_.destroySampler(textureSampler_, nullptr);
		device_.destroyImageView(textureImageView_, nullptr);

		device_.destroyImage(textureImage_, nullptr);
		device_.freeMemory(textureImageMemory_, nullptr);

		device_.destroyDescriptorSetLayout(descriptorSetLayout_, nullptr);

		device_.destroyBuffer(indexBuffer_, nullptr);
		device_.freeMemory(indexBufferMemory_, nullptr);

		device_.destroyBuffer(vertexBuffer_, nullptr);
		device_.freeMemory(vertexBufferMemory_, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			device_.destroySemaphore(renderFinishedSemaphores_[i], nullptr);
			device_.destroySemaphore(imageAvailableSemaphores_[i], nullptr);
			device_.destroyFence(inFlightFences_[i], nullptr);
		}

		device_.destroyCommandPool(commandPool_, nullptr);

		device_.destroy();

		instance_.destroySurfaceKHR(surface_, nullptr);

#ifdef ENABLE_VALIDATION_LAYERS
		DestroyDebugUtilsMessengerEXT(instance_, debugMessenger_, nullptr);
#endif

		instance_.destroy();

		glfwDestroyWindow(window_);

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
