#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 vertex_color;

uniform mat4 mvp;

out vec3 v_color;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    v_color = vertex_color;
}
