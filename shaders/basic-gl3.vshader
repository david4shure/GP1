#version 130

uniform mat4 uProjMatrix;
uniform mat4 uModelViewMatrix;

in vec3 aPosition;
in vec2 aTexCoord0; 

out vec3 vPosition;
out vec2 vTexCoord0;

void main() {
  // send position (eye coordinates) to fragment shader
  vec4 tPosition = uModelViewMatrix * vec4(aPosition, 1.0);
  vPosition = vec3(tPosition);
  gl_Position = uProjMatrix * tPosition;
  vTexCoord0 = aTexCoord0; // pass through texture coords
}
