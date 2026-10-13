#version 330 core

in vec2 v_uv;

uniform sampler2D tex;
uniform vec4 color;

out vec4 FragColor;

void main() {
    FragColor = texture(tex, v_uv) * color;
}
