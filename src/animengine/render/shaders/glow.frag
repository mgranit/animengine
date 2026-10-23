#version 330 core

uniform vec4 color;
uniform float core;
uniform float half_distance;
uniform float power;

in vec2 offset;
in vec3 tint;

out vec4 FragColor;

void main() {
    // Keep in sync with _GLOW_CUTOFF in material.py.
    const float cutoff = 1.0 / 255.0;

    float x = max(length(offset) - core, 0.0);
    float brightness = 1.0 / (1.0 + pow(x / half_distance, power));
    // Shift so the falloff reaches exactly zero at the sprite edge instead of a visible step.
    brightness = max(brightness - cutoff, 0.0) / (1.0 - cutoff);
    FragColor = vec4(color.rgb * tint, color.a * brightness);
}
