uniform float pointRadius;  // point size in world space
in vec4 clipPos;

void main()
{
    const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_TexCoord[0].xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0); //�ؽ�ó ��ǥ ���� ���� (-1~1)
    float mag = dot(N.xy, N.xy); //x^2 + y^2 ���� ������
    if (mag > 1.0) discard; // kill pixels outside circle
    N.z = sqrt(1.0f - mag); //�� ���������� ���� ���� ���
    normalize(N);

    // calculate depth
    float depth = clipPos.z / clipPos.w;
    gl_FragDepth = depth;

    // calculate lighting
    float diffuse = max(0.0, dot(lightDir, N));
    //gl_FragColor = gl_Color * diffuse;

    // visualize depth
    gl_FragColor = vec4(vec3(depth), 1.0);
}