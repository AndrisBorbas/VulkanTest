#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <vulkan/vulkan.hpp>

#include <vector>

vk::Instance createInstance(const std::vector<const char*>& validationLayers);

vk::DebugUtilsMessengerEXT setupDebugMessenger(vk::Instance& instance);

void DestroyDebugUtilsMessengerEXT(vk::Instance& instance,
								   vk::DebugUtilsMessengerEXT& debugMessenger,
								   const vk::AllocationCallbacks* pAllocator);

vk::SurfaceKHR createSurface(vk::Instance& instance, GLFWwindow* window);