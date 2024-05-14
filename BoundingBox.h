#ifndef __BOUNDINGBOX_H__
#define __BOUNDINGBOX_H__

#include "CUDA_Custom/DeviceManager.h"
struct OBB
{
	REAL _axis[3][3] = { {1,0,0},
						 {0,1,0},
						 {0,0,1} };
	REAL3 _center;
	REAL3 _center0;
	REAL3 _radius;
	REAL3 _corners[8];
};

static int OBB_NODE_QUAD_TABLE[6][4] =
{
   { 0, 4, 6, 2 },
   { 1, 3, 7, 5 },
   { 0, 1, 5, 4 },
   { 3, 2, 6, 7 },
   { 4, 5, 7, 6 },
   { 0, 2, 3, 1 }
};

static double OBB_NODE_NORMAL_TABLE[6][3] =
{
   { 0.0, 0.0, 1.0 },
   { 0.0, 0.0, -1.0 },
   { 0.0, 1.0, 0.0 },
   { 0.0, -1.0, 0.0 },
   { -1.0, 0.0, 0.0 },
   { 1.0, 0.0, 0.0 }
};

static int OBB_NODE_POSITIVE_TABLE[2][9] =
{
   { 0, 1, 1, 2, 2, 0, 2, 2, 1 },
   { 0, 1, 1, 2, 2, 2, 2, 2, 3 }
};

static int OBB_NODE_NEGATIVE_TABLE[2][9] =
{
   { 0, 1, 1, 2, 2, 4, 2, 2, 5 },
   { 0, 1, 1, 2, 2, 6, 2, 2, 7 }
};

static double __inline__ __host__ __device__ getDist(OBB& box, REAL3& pos)
{
    double diffs[3];
    bool allNegative = true;
    REAL radius[3] = { box._radius.x, box._radius.y, box._radius.z };
    for (int i = 0; i < 3; i++) {
        REAL3 axis = make_REAL3(box._axis[i][0], box._axis[i][1], box._axis[i][2]);
        diffs[i] = abs(Dot(axis, box._center - pos)) - radius[i];
        if (diffs[i] >= 0.0) {
            allNegative = false;
        }
    }
    if (allNegative) {
        double maxValue = diffs[0];
        for (int i = 1; i < 3; i++) {
            if (diffs[i] > maxValue) {
                maxValue = diffs[i];
            }
        }
        return maxValue;
    }
    double value = 0.0;
    for (int i = 0; i < 3; i++) {
        if (diffs[i] > 0.0) {
            value += diffs[i] * diffs[i];
        }
    }
    return sqrt(value);
}

static void  __inline__ __host__ computeCorners(OBB& box)
{
    REAL3 origin = box._center;
    REAL3 temp, axis;
    REAL3 Positive_Axis[3], negative_Axis[3];
    REAL radius[3] = { box._radius.x, box._radius.y, box._radius.z };
    for (int j = 0; j < 3; j++) {
        axis.x = box._axis[j][0];
        axis.y = box._axis[j][1];
        axis.z = box._axis[j][2];
        Normalize(axis);

        axis *= radius[j];
        Positive_Axis[j] = axis + origin;
        negative_Axis[j] = (Positive_Axis[j] - origin) * -1.0;
        negative_Axis[j] = negative_Axis[j] + origin;
    }
    for (int j = 0; j < 2; j++) {
        origin = Positive_Axis[OBB_NODE_POSITIVE_TABLE[j][0]];
        if (j == 0) {
            axis.x = box._axis[OBB_NODE_POSITIVE_TABLE[j][1]][0];
            axis.y = box._axis[OBB_NODE_POSITIVE_TABLE[j][1]][1];
            axis.z = box._axis[OBB_NODE_POSITIVE_TABLE[j][1]][2];
        }
        else {
            axis.x = -box._axis[OBB_NODE_POSITIVE_TABLE[j][1]][0];
            axis.y = -box._axis[OBB_NODE_POSITIVE_TABLE[j][1]][1];
            axis.z = -box._axis[OBB_NODE_POSITIVE_TABLE[j][1]][2];
        }
        Normalize(axis);
        axis *= radius[OBB_NODE_POSITIVE_TABLE[j][2]];
        temp = axis + origin;

        axis.x = box._axis[OBB_NODE_POSITIVE_TABLE[j][3]][0];
        axis.y = box._axis[OBB_NODE_POSITIVE_TABLE[j][3]][1];
        axis.z = box._axis[OBB_NODE_POSITIVE_TABLE[j][3]][2];
        Normalize(axis);
        axis *= radius[OBB_NODE_POSITIVE_TABLE[j][4]];
        box._corners[OBB_NODE_POSITIVE_TABLE[j][5]] = axis + temp;

        axis.x = -box._axis[OBB_NODE_POSITIVE_TABLE[j][6]][0];
        axis.y = -box._axis[OBB_NODE_POSITIVE_TABLE[j][6]][1];
        axis.z = -box._axis[OBB_NODE_POSITIVE_TABLE[j][6]][2];
        Normalize(axis);
        axis *= radius[OBB_NODE_POSITIVE_TABLE[j][7]];
        box._corners[OBB_NODE_POSITIVE_TABLE[j][8]] = axis + temp;
    }
    for (int j = 0; j < 2; j++) {
        origin = negative_Axis[OBB_NODE_NEGATIVE_TABLE[j][0]];
        if (j == 0) {
            axis.x = box._axis[OBB_NODE_NEGATIVE_TABLE[j][1]][0];
            axis.y = box._axis[OBB_NODE_NEGATIVE_TABLE[j][1]][1];
            axis.z = box._axis[OBB_NODE_NEGATIVE_TABLE[j][1]][2];
        }
        else {
            axis.x = -box._axis[OBB_NODE_NEGATIVE_TABLE[j][1]][0];
            axis.y = -box._axis[OBB_NODE_NEGATIVE_TABLE[j][1]][1];
            axis.z = -box._axis[OBB_NODE_NEGATIVE_TABLE[j][1]][2];
        }

        Normalize(axis);
        axis *= radius[OBB_NODE_NEGATIVE_TABLE[j][2]];
        temp = axis + origin;

        axis.x = box._axis[OBB_NODE_NEGATIVE_TABLE[j][3]][0];
        axis.y = box._axis[OBB_NODE_NEGATIVE_TABLE[j][3]][1];
        axis.z = box._axis[OBB_NODE_NEGATIVE_TABLE[j][3]][2];
        Normalize(axis);
        axis *= radius[OBB_NODE_NEGATIVE_TABLE[j][4]];
        box._corners[OBB_NODE_NEGATIVE_TABLE[j][5]] = axis + temp;

        axis.x = -box._axis[OBB_NODE_NEGATIVE_TABLE[j][6]][0];
        axis.y = -box._axis[OBB_NODE_NEGATIVE_TABLE[j][6]][1];
        axis.z = -box._axis[OBB_NODE_NEGATIVE_TABLE[j][6]][2];
        Normalize(axis);
        axis *= radius[OBB_NODE_NEGATIVE_TABLE[j][7]];
        box._corners[OBB_NODE_NEGATIVE_TABLE[j][8]] = axis + temp;
    }
}

static void   __inline__ __host__ Product(REAL a[][4], REAL b[][4], REAL c[][4])
{
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			REAL num = 0;
			for (int k = 0; k < 4; k++) {
				num += a[i][k] * b[k][j];
			}
			c[i][j] = num;
		}
	}
}

static void   __inline__ __host__ SetIdentity(REAL a[][4])
{
	a[0][0] = 1; a[0][1] = 0; a[0][2] = 0; a[0][3] = 0;
	a[1][0] = 0; a[1][1] = 1; a[1][2] = 0; a[1][3] = 0;
	a[2][0] = 0; a[2][1] = 0; a[2][2] = 1; a[2][3] = 0;
	a[3][0] = 0; a[3][1] = 0; a[3][2] = 0; a[3][3] = 1;
}

static void  __inline__ __host__  RotateMovingBox_kernel(OBB& box, bool type)
{
	box._center0 = box._center;

	REAL radian = type == true ? 1.5 : -1.5;
	REAL theta = radian * 0.017453292519943295769236907684886;
	REAL3 pivot = make_REAL3(0.5, 0.0, 0.5);
	REAL3 normal = make_REAL3(0.0, 1.0, 0.0);

	REAL rotate[4][4];
	REAL translate[4][4];
	REAL inv_translate[4][4];

	// initialize matrix
	SetIdentity(rotate);
	SetIdentity(translate);
	SetIdentity(inv_translate);

	// compute rotate matrix
	rotate[0][0] = cos(theta) + (normal.x * normal.x) * (1.0 - cos(theta));
	rotate[0][1] = (normal.x * normal.y) * (1.0 - cos(theta)) - normal.z * sin(theta);
	rotate[0][2] = (normal.x * normal.z) * (1.0 - cos(theta)) + normal.y * sin(theta);
	rotate[1][0] = (normal.y * normal.x) * (1.0 - cos(theta)) + normal.z * sin(theta);
	rotate[1][1] = cos(theta) + (normal.y * normal.y) * (1.0 - cos(theta));
	rotate[1][2] = (normal.y * normal.z) * (1.0 - cos(theta)) - normal.x * sin(theta);
	rotate[2][0] = (normal.z * normal.x) * (1.0 - cos(theta)) - normal.y * sin(theta);
	rotate[2][1] = (normal.z * normal.y) * (1.0 - cos(theta)) + normal.x * sin(theta);
	rotate[2][2] = cos(theta) + (normal.z * normal.z) * (1.0 - cos(theta));

	// compute translate matrix
	translate[0][3] = pivot.x; translate[1][3] = pivot.y; translate[2][3] = pivot.z;
	inv_translate[0][3] = -pivot.x; inv_translate[1][3] = -pivot.y; inv_translate[2][3] = -pivot.z;

	double undeformed_position[4], deformed_position[4];
	REAL transform1[4][4];
	REAL transform2[4][4];

	Product(translate, rotate, transform1);
	Product(transform1, inv_translate, transform2);

	undeformed_position[0] = box._center.x;
	undeformed_position[1] = box._center.y;
	undeformed_position[2] = box._center.z;
	undeformed_position[3] = 1.0;

	// transformation matrix
	for (int j = 0; j < 4; j++) {
		deformed_position[j] = 0.0;
		for (int k = 0; k < 4; k++) {
			deformed_position[j] += transform2[j][k] * undeformed_position[k];
		}
	}

	// update position
	box._center = make_REAL3(deformed_position[0], deformed_position[1], deformed_position[2]);


	// 2. box center를 기준으로 rotate      
	theta *= -1.0;
	pivot = box._center;
	pivot.y = 0.5f;

	// initialize matrix
	SetIdentity(rotate);
	SetIdentity(translate);
	SetIdentity(inv_translate);

	// compute rotate matrix
	rotate[0][0] = cos(theta) + (normal.x * normal.x) * (1.0 - cos(theta));
	rotate[0][1] = (normal.x * normal.y) * (1.0 - cos(theta)) - normal.z * sin(theta);
	rotate[0][2] = (normal.x * normal.z) * (1.0 - cos(theta)) + normal.y * sin(theta);
	rotate[1][0] = (normal.y * normal.x) * (1.0 - cos(theta)) + normal.z * sin(theta);
	rotate[1][1] = cos(theta) + (normal.y * normal.y) * (1.0 - cos(theta));
	rotate[1][2] = (normal.y * normal.z) * (1.0 - cos(theta)) - normal.x * sin(theta);
	rotate[2][0] = (normal.z * normal.x) * (1.0 - cos(theta)) - normal.y * sin(theta);
	rotate[2][1] = (normal.z * normal.y) * (1.0 - cos(theta)) + normal.x * sin(theta);
	rotate[2][2] = cos(theta) + (normal.z * normal.z) * (1.0 - cos(theta));

	// compute translate matrix
	translate[0][3] = pivot.x; translate[1][3] = pivot.y; translate[2][3] = pivot.z;
	inv_translate[0][3] = -pivot.x; inv_translate[1][3] = -pivot.y; inv_translate[2][3] = -pivot.z;

	// compute transformation
	SetIdentity(transform1);
	SetIdentity(transform2);

	Product(translate, rotate, transform1);
	Product(transform1, inv_translate, transform2);

	// transformation matrix
	REAL a[3][3];
	REAL self_rotate[3][3];
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			self_rotate[i][j] = transform2[i][j];
			a[i][j] = box._axis[i][j];
		}
	}

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			REAL num = 0;
			for (int k = 0; k < 3; k++) {
				num += a[i][k] * self_rotate[k][j];
			}
			box._axis[i][j] = num;
		}
	}
	computeCorners(box);
}
#endif
