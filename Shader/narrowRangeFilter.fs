#version 410 core

#define FIX_OTHER_WEIGHT
#define RANGE_EXTENSION

#define PI_OVER_8 0.392699082f

uniform sampler2D depthMap;
uniform float     pointRadius;
uniform int       filterRadius;
uniform int       maxFilterRadius;
uniform int       screenWidth;
uniform int       screenHeight;

// doFilter1D = 1, 0, and -1 (-1 mean filter2D with fixed radius)
uniform int doFilter1D;
uniform int blurDir;

in vec2   coord;

const int   fixedFilterRadius = 5;
const float thresholdRatio    = 10.5;
const float clampRatio        = 0.5;
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

float Gaussian1D(float r, float two_sigma2)
{
    return exp(-r * r / two_sigma2);
}

float Gaussian2D(vec2 r, float two_sigma2)
{
    return exp(-dot(r, r) / two_sigma2);
}

void ModifiedGaussianFilter(inout float sampleDepth, inout float weight, inout float weight_other, inout float upper, inout float lower, float lower_clamp, float threshold)
{
    if(sampleDepth > upper) {
        weight = 0;
        weight_other = 0;
    } else {
        if(sampleDepth < lower) {
            sampleDepth = lower_clamp;
        }
        else {
            upper = max(upper, sampleDepth + threshold);
            lower = min(lower, sampleDepth - threshold);
        }
    }
}



float filter1D(float pixelDepth)
{
    if(filterRadius == 0) {
        return pixelDepth;
    }

    vec2  blurRadius = vec2(1.0 / screenWidth, 1.0 / screenHeight);
    float threshold  = pointRadius * thresholdRatio;
    float ratio      = screenHeight / 2.0 / tan(PI_OVER_8);
    float K          = -filterRadius * ratio * pointRadius * 0.1f;
    int   filterSize = min(maxFilterRadius, int(ceil(K / pixelDepth)));

    float upper       = pixelDepth + threshold;
    float lower       = pixelDepth - threshold;
    float lower_clamp = pixelDepth - pointRadius * clampRatio;

    float sigma      = filterSize / 3.0f;
    float two_sigma2 = 2.0f * sigma * sigma;

    vec2 sum2  = vec2(pixelDepth, 0);
    vec2 wsum2 = vec2(1, 0);
    vec4 dir   = (blurDir == 0) ? vec4(blurRadius.x, 0, -blurRadius.x, 0) : vec4(0, blurRadius.y, 0, -blurRadius.y);

    vec4  sampleCoord = coord.xyxy;
    float r     = 0;
    float dr    = dir.x + dir.y;

    float upper1 = upper;
    float upper2 = upper;
    float lower1 = lower;
    float lower2 = lower;
    vec2  sampleDepth;
    vec2  w2;

    for(int x = 1; x <= filterSize; ++x) {
        sampleCoord += dir;
        r     += dr;

        sampleDepth.x = texture(depthMap, sampleCoord.xy).r;
        sampleDepth.y = texture(depthMap, sampleCoord.zw).r;
        sampleDepth.x = NormalizedDepthToDepth(sampleDepth.x);
        sampleDepth.y = NormalizedDepthToDepth(sampleDepth.y);

        w2 = vec2(Gaussian1D(r, two_sigma2));
        ModifiedGaussianFilter(sampleDepth.x, w2.x, w2.y, upper1, lower1, lower_clamp, threshold);
        ModifiedGaussianFilter(sampleDepth.y, w2.y, w2.x, upper2, lower2, lower_clamp, threshold);

        sum2  += sampleDepth * w2;
        wsum2 += w2;
    }

    vec2 filterVal = vec2(sum2.x, wsum2.x) + vec2(sum2.y, wsum2.y);
    return filterVal.x / filterVal.y;
}

float filter2D(float pixelDepth)
{
    if(filterRadius == 0) {
        return pixelDepth;
    }

    vec2  blurRadius = vec2(1.0 / screenWidth, 1.0 / screenHeight);
    float threshold  = pointRadius * thresholdRatio;
    float ratio      = screenHeight / 2.0 / tan(PI_OVER_8);
    float K          = -filterRadius * ratio * pointRadius * 0.1f;
    int   filterSize = (doFilter1D < 0) ? fixedFilterRadius : min(maxFilterRadius, int(ceil(K / pixelDepth)));

    float upper       = pixelDepth + threshold;
    float lower       = pixelDepth - threshold;
    float lower_clamp = pixelDepth - pointRadius * clampRatio;

    float sigma      = filterSize / 3.0f;
    float two_sigma2 = 2.0f * sigma * sigma;

    vec4 sampleCoord = coord.xyxy;

    vec2 r     = vec2(0, 0);
    vec4 sum4  = vec4(pixelDepth, 0, 0, 0);
    vec4 wsum4 = vec4(1, 0, 0, 0);
    vec4 sampleDepth;
    vec4 w4;

    for(int x = 1; x <= filterSize; ++x) {
        r.x     += blurRadius.x;
        sampleCoord.x += blurRadius.x;
        sampleCoord.z -= blurRadius.x;
        vec4 sampleCoord1 = sampleCoord.xyxy;
        vec4 sampleCoord2 = sampleCoord.zwzw;

        for(int y = 1; y <= filterSize; ++y) {
            sampleCoord1.y += blurRadius.y;
            sampleCoord1.w -= blurRadius.y;
            sampleCoord2.y += blurRadius.y;
            sampleCoord2.w -= blurRadius.y;

            sampleDepth.x = texture(depthMap, sampleCoord1.xy).x;
            sampleDepth.y = texture(depthMap, sampleCoord1.zw).x;
            sampleDepth.z = texture(depthMap, sampleCoord2.xy).x;
            sampleDepth.w = texture(depthMap, sampleCoord2.zw).x;

            sampleDepth.x = NormalizedDepthToDepth(sampleDepth.x);
            sampleDepth.y = NormalizedDepthToDepth(sampleDepth.y);
            sampleDepth.z = NormalizedDepthToDepth(sampleDepth.z);
            sampleDepth.w = NormalizedDepthToDepth(sampleDepth.w);

            r.y += blurRadius.y;
            w4   = vec4(Gaussian2D(blurRadius * r, two_sigma2));

            ModifiedGaussianFilter(sampleDepth.x, w4.x, w4.w, upper, lower, lower_clamp, threshold);
            ModifiedGaussianFilter(sampleDepth.y, w4.y, w4.z, upper, lower, lower_clamp, threshold);
            ModifiedGaussianFilter(sampleDepth.z, w4.z, w4.y, upper, lower, lower_clamp, threshold);
            ModifiedGaussianFilter(sampleDepth.w, w4.w, w4.x, upper, lower, lower_clamp, threshold);

            sum4  += sampleDepth * w4;
            wsum4 += w4;
        }
    }

    vec2 filterVal;
    filterVal.x = dot(sum4, vec4(1, 1, 1, 1));
    filterVal.y = dot(wsum4, vec4(1, 1, 1, 1));
    return filterVal.x / filterVal.y;
}

void main()
{
    float pixelDepth = texture(depthMap, coord).x;
    float depth = NormalizedDepthToDepth(pixelDepth);

    if(depth > 0.0 || depth < -1000.0f) {
        gl_FragDepth = DepthToNormalizedDepth(depth);
    } else {
        depth = (doFilter1D == 1) ? filter1D(depth) : filter2D(depth);
        gl_FragDepth = DepthToNormalizedDepth(depth);
    }

}