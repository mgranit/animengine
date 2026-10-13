#version 330 core

uniform vec4 color;

out vec4 FragColor;

void main() {
    if (length(gl_PointCoord - vec2(0.5)) > 0.5) {
        discard;
    }
    FragColor = color;
}
