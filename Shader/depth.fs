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
    normal.xy = gl_PointCoord * 2.0 - 1.0; //�ؽ�ó ��ǥ ���� ���� (-1~1)
    float mag = dot(normal.xy, normal.xy); //x^2 + y^2 ���� ������
    if (mag > 1.0) discard; // kill pixels outside circle
    normal.z = sqrt(1.0f - mag); //�� ���������� ���� ���� ���

    // calculate depth
    vec4 pixelPos = vec4(pos + normal * pointRadius, 1.0);
    vec4 clipSpacePos = projection * pixelPos;

    float depth = (clipSpacePos.z / clipSpacePos.w) * 0.5f + 0.5f;
    gl_FragDepth = depth;

    // calculate lighting
    //float diffuse = max(0.0, dot(lightDir, N));
    //gl_FragColor = gl_Color * diffuse;

    // visualize depth
    //fragColor = vec4(vec3(depth), 1.0);
}