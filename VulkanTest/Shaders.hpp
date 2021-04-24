#pragma once

#include <vulkan/vulkan.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <array>
#include <glm/glm.hpp>
#include <vector>

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;

	static vk::VertexInputBindingDescription getBindingDescription()
	{
		vk::VertexInputBindingDescription bindingDescription;
		bindingDescription.binding   = 0;
		bindingDescription.stride    = sizeof(Vertex);
		bindingDescription.inputRate = vk::VertexInputRate::eVertex;

		return bindingDescription;
	}

	static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions()
	{
		std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions;

		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].binding  = 0;
		attributeDescriptions[0].format   = vk::Format::eR32G32B32Sfloat;
		attributeDescriptions[0].offset   = offsetof(Vertex, pos);

		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].binding  = 0;
		attributeDescriptions[1].format   = vk::Format::eR32G32B32Sfloat;
		attributeDescriptions[1].offset   = offsetof(Vertex, color);

		attributeDescriptions[2].binding  = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format   = vk::Format::eR32G32Sfloat;
		attributeDescriptions[2].offset   = offsetof(Vertex, texCoord);

		return attributeDescriptions;
	}

	bool operator==(const Vertex& other) const
	{
		return pos == other.pos && color == other.color && texCoord == other.texCoord;
	}
};

namespace std
{
template <>
struct hash<Vertex>
{
	size_t operator()(Vertex const& vertex) const
	{
		return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1)
			   ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
	}
};
}  // namespace std

struct UniformBufferObject
{
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

uint32_t findMemoryType(vk::PhysicalDevice physicalDevice,
						uint32_t typeFilter,
						vk::MemoryPropertyFlags properties);
