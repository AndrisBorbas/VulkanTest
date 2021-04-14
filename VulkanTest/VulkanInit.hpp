#pragma once

#include <vulkan/vulkan.hpp>

#include <vector>

vk::Instance createInstance(const std::vector<const char*>& validationLayers);

vk::DebugUtilsMessengerEXT setupDebugMessenger(vk::Instance& instance);

void DestroyDebugUtilsMessengerEXT(vk::Instance& instance,
								   vk::DebugUtilsMessengerEXT& debugMessenger,
								   const vk::AllocationCallbacks* pAllocator);
