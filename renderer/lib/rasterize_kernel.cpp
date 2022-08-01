/*
 Author: Yao Feng
 Modified by Jianzhu Guo

 functions that can not be optimazed by vertorization in python.
 1. rasterization.(need process each triangle)
 2. normal of each vertex.(use one-ring, need process each vertex)
 3. write obj(seems that it can be verctorized? anyway, writing it in c++ is simple, so also add function here. --> however, why writting in c++ is still slow?)



*/

#include "rasterize.h"
#include<omp.h>


void get_point_weight(float *weight, Point p, Point p0, Point p1, Point p2) {
    // vectors
    Point v0, v1, v2;
    v0 = p2 - p0;
    v1 = p1 - p0;
    v2 = p - p0;

    // dot products
    float dot00 = v0.dot(v0); //v0.x * v0.x + v0.y * v0.y //np.dot(v0.T, v0)
    float dot01 = v0.dot(v1); //v0.x * v1.x + v0.y * v1.y //np.dot(v0.T, v1)
    float dot02 = v0.dot(v2); //v0.x * v2.x + v0.y * v2.y //np.dot(v0.T, v2)
    float dot11 = v1.dot(v1); //v1.x * v1.x + v1.y * v1.y //np.dot(v1.T, v1)
    float dot12 = v1.dot(v2); //v1.x * v2.x + v1.y * v2.y//np.dot(v1.T, v2)

    // barycentric coordinates
    float inverDeno;
    if (dot00 * dot11 - dot01 * dot01 == 0)
        inverDeno = 0;
    else
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);

    float u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
    float v = (dot00 * dot12 - dot01 * dot02) * inverDeno;

    // weight
    weight[0] = 1 - u - v;
    weight[1] = v;
    weight[2] = u;
}

// rasterization by Z-Buffer with optimization
// Complexity: < ntri * h * w * c
void _rasterize(
        unsigned char *image, float *vertices, int *triangles, float *colors, float *depth_buffer,
        int ntri, int h, int w) {
    const int c = 3;

    #pragma omp parallel for
    for (int i = 0; i < ntri; i++) {
        int tri_p0_ind, tri_p1_ind, tri_p2_ind;
        Point p0, p1, p2, p;
        int x_min, x_max, y_min, y_max;
        float p_depth, p0_depth, p1_depth, p2_depth;
        float p_color, p0_color, p1_color, p2_color;
        float weight[3];
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        p0.x = vertices[3 * tri_p0_ind];
        p0.y = vertices[3 * tri_p0_ind + 1];
        p0_depth = vertices[3 * tri_p0_ind + 2];
        p1.x = vertices[3 * tri_p1_ind];
        p1.y = vertices[3 * tri_p1_ind + 1];
        p1_depth = vertices[3 * tri_p1_ind + 2];
        p2.x = vertices[3 * tri_p2_ind];
        p2.y = vertices[3 * tri_p2_ind + 1];
        p2_depth = vertices[3 * tri_p2_ind + 2];

        x_min = max((int) ceil(min(p0.x, min(p1.x, p2.x))), 0);
        x_max = min((int) floor(max(p0.x, max(p1.x, p2.x))), w - 1);

        y_min = max((int) ceil(min(p0.y, min(p1.y, p2.y))), 0);
        y_max = min((int) floor(max(p0.y, max(p1.y, p2.y))), h - 1);

        if (x_max < x_min || y_max < y_min) {
            continue;
        }

        for (int y = y_min; y <= y_max; y++) {
            for (int x = x_min; x <= x_max; x++) {
                p.x = float(x);
                p.y = float(y);

                // call get_point_weight function once
                get_point_weight(weight, p, p0, p1, p2);

                // and judge is_point_in_tri by below line of code
                if (weight[2] > 0 && weight[1] > 0 && weight[0] > 0) {
                    p_depth = weight[0] * p0_depth + weight[1] * p1_depth + weight[2] * p2_depth;

                    if ((p_depth > depth_buffer[y * w + x])) {
                        depth_buffer[y * w + x] = p_depth;
                        for (int k = 0; k < c; k++) {
                            p0_color = colors[c * tri_p0_ind + k];
                            p1_color = colors[c * tri_p1_ind + k];
                            p2_color = colors[c * tri_p2_ind + k];

                            p_color = weight[0] * p0_color + weight[1] * p1_color + weight[2] * p2_color;
                            image[(h - 1 - y) * w * c + x * c + k] = (unsigned char) (255 * p_color);
                        }

                    }
                }
            }
        }
    }
}

