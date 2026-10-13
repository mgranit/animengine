#version 330 core

layout (location = 0) in vec3 position;

uniform mat4 mvp;
uniform float point_size;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    gl_PointSize = point_size;
}
