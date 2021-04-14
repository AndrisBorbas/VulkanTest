#pragma once

#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>

#include <optional>
#include <set>
#include <vector>

vk::ShaderModule createShaderModule(vk::Device device, const std::vector<char>& code)
{
	vk::ShaderModuleCreateInfo createInfo;
	createInfo.codeSize = code.size();
	createInfo.pCode    = reinterpret_cast<const uint32_t*>(code.data());
	vk::ShaderModule shaderModule;
	if (device.createShaderModule(&createInfo, nullptr, &shaderModule) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create shader module!");
	}
	return shaderModule;
}
