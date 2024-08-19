uniform sampler2D depthTex;  // depth �ؽ�ó
uniform float maxDepth;
uniform vec2 screenSize;

void main() {
    vec2 texCoord = gl_FragCoord.xy / screenSize;  // �ؽ�ó ��ǥ
    float depth = texture2D(depthTex, texCoord).x;  // ���� �� ���ø�

    if (depth > maxDepth)
        discard;  // ���̰� maxDepth�� �ʰ��ϸ� �ȼ��� ����

    // ���� ���� �������� ��ȯ�Ͽ� ���
    gl_FragColor = vec4(vec3(depth), 1.0);
}