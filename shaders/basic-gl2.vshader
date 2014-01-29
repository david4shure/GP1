uniform mat4 uProjMatrix;
uniform mat4 uModelViewMatrix;

attribute vec3 aPosition;
attribute vec2 aTexCoord0;

varying vec3 vPosition;
varying vec2 vTexCoord0;

void main() {
  // send position (eye coordinates) to fragment shader
  vec4 tPosition = uModelViewMatrix * vec4(aPosition, 1.0);
  vPosition = vec3(tPosition);
  gl_Position = uProjMatrix * tPosition;
  vTexCoord0 = aTexCoord0; // pass through texture coords
}
