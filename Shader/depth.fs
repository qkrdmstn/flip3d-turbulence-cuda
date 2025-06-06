#version 400 core

in vec3 pos;

uniform mat4 mView;
uniform mat4 projection;
uniform float pointRadius;

//out vec4 fragColor;
void main()
{
    // calculate normal from texture coordinates
    vec3 normal;
    normal.xy = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0); //텍스처 좌표 범위 변경 (-1~1)
    float mag = dot(normal.xy, normal.xy); //x^2 + y^2 원의 방정식
    if (mag > 1.0) discard; // kill pixels outside circle
    normal.z = sqrt(1.0f - mag); //구 방정식으로 법선 벡터 계산
    
    //// calculate depth
    vec4 pixelPos = vec4(pos + normal * pointRadius, 1.0);
    vec4 clipSpacePos = projection * pixelPos;
    
    float depth = (clipSpacePos.z / clipSpacePos.w);
    gl_FragDepth = depth;
}