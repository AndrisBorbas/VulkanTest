#pragma once

#define NOMINMAX

#define VULKAN_HPP_NO_NODISCARD_WARNINGS

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS

#define VK_USE_PLATFORM_WIN32_KHR

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <vulkan/vulkan.hpp>
