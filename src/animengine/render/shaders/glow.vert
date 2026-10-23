#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 vertex_color;

uniform mat4 mvp;
uniform float extent;

out vec2 offset;
out vec3 tint;

void main() {
    // One instance per vertex; gl_VertexID 0..3 are the corners of a triangle strip.
    vec2 corner = vec2(gl_VertexID & 1, gl_VertexID >> 1) * 2.0 - 1.0;
    offset = corner * extent;
    tint = vertex_color;
    gl_Position = mvp * vec4(position + vec3(offset, 0.0), 1.0);
}
