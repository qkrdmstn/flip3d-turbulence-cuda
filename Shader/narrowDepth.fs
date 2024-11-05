#version 400 core

in vec3 pos;

uniform mat4 mView;
uniform mat4 projection;
uniform float pointRadius;

const float farZ  = 1000.0f;
const float nearZ = 0.1f;

float DepthToNormalizedDepth(float z) 
{
    float a = (farZ + nearZ) / (farZ - nearZ);
    float b = 2.0 * farZ * nearZ / (farZ - nearZ);
    float normalizedDepth = (a * z + b) / z;

    return normalizedDepth;
}

float NormalizedDepthToDepth(float zn)
{
    float a = (farZ + nearZ) / (farZ - nearZ);
    float b = 2.0 * farZ * nearZ / (farZ - nearZ);
    float depth = b / (zn - a);

    return depth;
}

//out vec4 fragColor;
void main()
{
    // calculate normal from texture coordinates
    vec3 normal;
    normal.xy = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0); //�ؽ�ó ��ǥ ���� ���� (-1~1)
    float mag = dot(normal.xy, normal.xy); //x^2 + y^2 ���� ������
    if (mag > 1.0) discard; // kill pixels outside circle
    normal.z = sqrt(1.0f - mag); //�� ���������� ���� ���� ���

    // calculate depth
    vec4 pixelPos = vec4(pos + normal * pointRadius, 1.0);
    gl_FragDepth =  DepthToNormalizedDepth(pixelPos.z);
}