uniform sampler2D depthTex;  // depth 텍스처
uniform float maxDepth;
uniform vec2 screenSize;

void main() {
    vec2 texCoord = gl_FragCoord.xy / screenSize;  // 텍스처 좌표
    float depth = texture2D(depthTex, texCoord).x;  // 깊이 값 샘플링

    if (depth > maxDepth)
        discard;  // 깊이가 maxDepth를 초과하면 픽셀을 버림

    // 깊이 값을 색상으로 변환하여 출력
    gl_FragColor = vec4(vec3(depth), 1.0);
}