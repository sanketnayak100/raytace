#ifndef ray_funcs_h
#define ray_funcs_h

#include <iostream>
#include <cuda.h>
#include <thread>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <threads.h>

#include <tira/parser.h>
#include <tira/graphics/camera.h>
#include <tira/graphics/shapes/simplemesh.h>

#include <tira/cuda/error.h>
#include <tira/cuda/callable.h>

#ifndef __CUDACC__
#include "tira/image.h"
#endif

class spheres
{
public:
    glm::vec3 cen;
    float rad, r, g, b;
};

class lights
{
public:
    glm::vec3 pos;
    float r, g, b;
};

class camera
{
public:
    float pos, look, up, fov;
};

class ray
{
public:
    glm::vec3 dir, origin;
};

class plane
{
public:
    glm::vec3 point, normal;
    float r, g, b;
};

extern int res_x, res_y, num_sph, num_lights, num_planes, num_mesh;
extern tira::camera cam;
extern tira::simplemesh mesh;

extern spheres s[100], bound_sph;
extern lights l[100];
extern plane p;

__device__ inline CUDA_CALLABLE bool test_intersect(glm::vec3 origin, glm::vec3 dir, spheres sph, float& t)
{
    glm::vec3 l1;
    float s, l_sq, r_sq, m_sq, q;
    l1 = sph.cen - origin;
    s = glm::dot(l1, dir);
    l_sq = glm::dot(l1, l1);
    r_sq = glm::dot(sph.rad, sph.rad);
    if (s < 0 && l_sq < r_sq)
        return false;
    m_sq = l_sq - s * s;
    if (m_sq > r_sq)
        return false;
    if (l_sq < r_sq)
        return false;
    q = sqrt(r_sq - m_sq);
    t = s - q;
    return true;
}

__device__ inline CUDA_CALLABLE glm::vec3 plane_intersect(plane p, ray r)
{
    glm::vec3 point;
    float a, b, c, t;
    float x, y, z, d;
    a = glm::dot(p.normal, (p.point - r.origin));
    b = glm::dot(p.normal, r.dir);
    t = a / b;
    point = r.origin + t * r.dir;
    return point;
}

__device__ inline CUDA_CALLABLE bool test_shadow(glm::vec3 point, glm::vec3 dir, spheres* s, int num_sph, lights light, float& t)
{
    int k;
    for (k = 0; k < num_sph; k++)
    {
        if (test_intersect(point, dir, s[k], t) && t > 0)
            return true;
    }
    return false;
}

__device__ inline CUDA_CALLABLE static glm::vec3 lighting(ray r, spheres* sph, int num_sph, int sph_id, float t, lights* light, int num_lights)
{
    glm::vec3 p, d, n;

    float i_r = 0, i_g = 0, i_b = 0, i_temp, x = 0;
    int u, v;
    glm::vec3 col;
    p = r.origin + r.dir * t;
    n = (p - sph[sph_id].cen) / glm::length(p - sph[sph_id].cen);
    for (u = 0; u < num_lights; u++)
    {
        d = (light[u].pos - p) / glm::length(light[u].pos - p);
        i_temp = std::max(float(0), glm::dot(d / glm::length(d), n));
        if (test_shadow(p, d, sph, num_sph, light[u], x))
            i_temp = 0;
        /*for(v=0;v<num_sph;v++)
            if (test_intersect(p, d, sph[v], t) && t>0)
                i_temp = 0;*/
        i_r = i_r + i_temp * light[u].r;
        if (i_r > 1)
            i_r = 1;
        i_g = i_g + i_temp * light[u].g;
        if (i_g > 1)
            i_g = 1;
        i_b = i_b + i_temp * light[u].b;
        if (i_b > 1)
            i_b = 1;
    }

    col[0] = sph[sph_id].r * 255 * i_r;
    col[1] = sph[sph_id].g * 255 * i_g;
    col[2] = sph[sph_id].b * 255 * i_b;
    return col;
}

__device__ inline CUDA_CALLABLE static float test_triangle(tira::triangle tri, glm::vec3 point)
{
    glm::vec3 a, b, c;
    float test1, test2, test3;
    a = glm::cross(tri.v[1] - tri.v[0], point - tri.v[0]);
    b = glm::cross(tri.v[2] - tri.v[1], point - tri.v[1]);
    c = glm::cross(tri.v[0] - tri.v[2], point - tri.v[2]);
    test1 = glm::dot(a, tri.n);
    test2 = glm::dot(b, tri.n);
    test3 = glm::dot(c, tri.n);
    if (test1 >= 0 && test2 >= 0 && test3 >= 0)
        return 1;
    else if (test1 <= 0 && test2 <= 0 && test3 <= 0)
        return 1;
    else
        return 0;
}

#ifdef __CUDACC__

__device__ inline CUDA_CALLABLE static glm::vec3 lighting_cuda(ray r, spheres* sph, int num_sph, int sph_id, float t, lights* light, int num_lights)
{
    glm::vec3 p, d, n;

    float i_r = 0, i_g = 0, i_b = 0, i_temp, x = 0;
    int u, v;
    glm::vec3 col;
    p = r.origin + r.dir * t;
    n = (p - sph[sph_id].cen) / glm::length(p - sph[sph_id].cen);
    for (u = 0; u < num_lights; u++)
    {
        d = (light[u].pos - p) / glm::length(light[u].pos - p);
        i_temp = glm::dot(d / glm::length(d), n);
        if (i_temp <0 || test_shadow(p, d, sph, num_sph, light[u], x))
            i_temp = 0;
        /*for(v=0;v<num_sph;v++)
            if (test_intersect(p, d, sph[v], t) && t>0)
                i_temp = 0;*/
        i_r = i_r + i_temp * light[u].r;
        if (i_r > 1)
            i_r = 1;
        i_g = i_g + i_temp * light[u].g;
        if (i_g > 1)
            i_g = 1;
        i_b = i_b + i_temp * light[u].b;
        if (i_b > 1)
            i_b = 1;
    }

    col[0] = sph[sph_id].r * 255 * i_r;
    col[1] = sph[sph_id].g * 255 * i_g;
    col[2] = sph[sph_id].b * 255 * i_b;
    return col;
}

__device__ inline CUDA_CALLABLE float test_intersect_cuda(glm::vec3 origin, glm::vec3 dir, spheres sph)
{
    glm::vec3 l1;
    float s, l_sq, r_sq, m_sq, q, t;
    l1 = sph.cen - origin;
    s = glm::dot(l1, dir);
    l_sq = glm::dot(l1, l1);
    r_sq = glm::dot(sph.rad, sph.rad);
    if (s < 0 && l_sq < r_sq)
        return -1;
    m_sq = l_sq - s * s;
    if (m_sq > r_sq)
        return -1;
    if (l_sq < r_sq)
        return -1;
    q = sqrtf(fmaxf(0.0f, r_sq - m_sq));
    t = s - q;
    if (t < 0)
        return -1;
    return t;
}

__global__ void raytrace_loop_cuda(tira::camera *cam, tira::triangle *tri1, spheres* s, spheres *bound_sph, lights* l, plane *p, int res_x, int res_y, int tri_count, int num_sph, int num_lights, int num_planes, int num_mesh, float* image_mat)
{
    int flag, k, u, min_k = 9999;
    float t_min, min_dist, shadow_r, shadow_g, shadow_b, color_r, color_g, color_b, shadow, dist, color, t;
    glm::vec3 plane_int, test, c, tri_int;
    double i, j;
    double i2, j2;
    //int i1, j1;
    ray pixel_ray;
    plane tri_p;
    tira::triangle tri;
    /*for (i = i_min - 0.5; i <= i_min - 0.4 ; i = i + 0.001)
        for (j = j_min - 0.5; j <= j_min - 0.4; j = j + 0.001)*/
    //for (i1 = 0; i1 < res_x ; i1++)
    //    for (j1 = 0; j1 < res_y; j1++)
    //    {
            //i = (i1 / res_x) - 0.5;
            //j = (j1 / res_y) - 0.5;
    size_t i1 = blockIdx.y * blockDim.y + threadIdx.y; 
    size_t j1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i1 > res_x || j1 > res_y)
        return;
    else
    {
        t = 0;
        i = (static_cast<double>(i1) / res_x) - 0.5;
        j = (static_cast<double>(j1) / res_y) - 0.5;
        flag = 0;
        t_min = 99999;
        shadow_r = 0;
        shadow_g = 0;
        shadow_b = 0;
        color_r = 0;
        color_g = 0;
        color_b = 0;
        shadow = 0;
        min_dist = 9999;
        pixel_ray.dir = cam->ray(i, j);
        pixel_ray.origin = cam->position();
        //    //test_intersect(pixel_ray, s, t);
        for (k = 0; k < num_sph; k++)
        {
            t = test_intersect_cuda(pixel_ray.origin, pixel_ray.dir, s[k]);
            if (t > 0 && t < t_min)
            {
                c = lighting_cuda(pixel_ray, s, num_sph, k, t, l, num_lights);
                t_min = t;
                flag = 1;
                image_mat[i1 * res_y * 3 + j1 * 3 + 0] = c.r;
                image_mat[i1 * res_y * 3 + j1 * 3 + 1] = c.g;
                image_mat[i1 * res_y * 3 + j1 * 3 + 2] = c.b;

            }
            
        }
        if (flag == 0 && num_planes)       //if no sphere intersections, and the ray is not parallel to the plane
        {
            plane_int = plane_intersect(*p, pixel_ray);
            /*if(int(i*10)==-4)
                std::cout <<"\nhi"<< plane_int.x << plane_int.y << plane_int.z<<"\n";*/
            for (u = 0; u < num_lights; u++)
            {
                //shadow = glm::dot(p.normal, pixel_ray.dir) / (glm::length(p.normal) * glm::length(pixel_ray.dir));
                shadow = 1;
                test = (plane_int - l[u].pos) / glm::length(plane_int - l[u].pos);
                //std::cout << "\n" << test.x << test.y << test.z;
                if (test_shadow(l[u].pos, test, s, num_sph, l[u], t))
                {
                    shadow = 0;
                }
                else
                {
                    //shadow = std::max(float(0),glm::dot(p.normal, -pixel_ray.dir) / (glm::length(p.normal) * glm::length(pixel_ray.dir)));
                    //shadow = std::max(float(0), glm::dot(((l[u].pos - plane_int) / glm::length(l[u].pos - plane_int)), p.normal));
                    shadow = glm::dot(((l[u].pos - plane_int) / glm::length(l[u].pos - plane_int)), p->normal);
                    if (shadow < 0)
                        shadow = 0;
                    //std::cout <<"\n"<< shadow;
                }
                shadow_r = shadow_r + shadow * 255 * l[u].r;
                shadow_g = shadow_g + shadow * 255 * l[u].g;
                shadow_b = shadow_b + shadow * 255 * l[u].b;
            }

            image_mat[i1 * res_y * 3 + j1 * 3 + 0] = p->r + shadow_r;
            image_mat[i1 * res_y * 3 + j1 * 3 + 1] = p->g + shadow_g;
            image_mat[i1 * res_y * 3 + j1 * 3 + 2] = p->b + shadow_b;
            //img(res_x - (i + 0.5) * res_x - 1, res_y - (j + 0.5) * res_y - 1, 0) = p.r * shadow_r;
            //img(res_x - (i + 0.5) * res_x - 1, res_y - (j + 0.5) * res_y - 1, 1) = p.g * shadow_g;
            //img(res_x - (i + 0.5) * res_x - 1, res_y - (j + 0.5) * res_y - 1, 2) = p.b * shadow_b;
        }
        
        if (num_mesh > 0 && test_intersect_cuda(pixel_ray.origin, pixel_ray.dir, *bound_sph) > 0)  //test intersect on bounding sphere
        {
            for (k = 0; k < num_mesh; k++)
            {
                tri = tri1[k];
                tri_p.point = tri.v[0];
                tri_p.normal = tri.n;
                tri_int = plane_intersect(tri_p, pixel_ray);
                if (test_triangle(tri, tri_int))
                {
                    dist = glm::distance(tri_int, cam->position());
                    if (dist < min_dist)
                    {
                        for (u = 0; u < num_lights; u++)
                        {
                            if (u == 0)
                            {
                                color_r = 0;
                                color_g = 0;
                                color_b = 0;
                            }
                            //color = std::max(float(0), glm::dot(((l[u].pos - tri_int) / glm::length(l[u].pos - tri_int)), tri_p.normal));
                            color = glm::dot(((l[u].pos - tri_int) / glm::length(l[u].pos - tri_int)), tri_p.normal);
                            if (color < 0)
                                color = 0;
                            color_r = color_r + color * l[u].r;
                            color_g = color_g + color * l[u].g;
                            color_b = color_b + color * l[u].b;
                            if (color_r > 1)
                                color_r = 1;
                            if (color_g > 1)
                                color_g = 1;
                            if (color_b > 1)
                                color_b = 1;
                        }
                        min_k = k;
                        min_dist = dist;
                    }
                }
            }
            if (min_k != -1)
            {
                //color = 1;
                //img(res_x - (i + 0.5) * res_x - 1, res_y - (j + 0.5) * res_y - 1, 0) = color_r * 255;
                //img(res_x - (i + 0.5) * res_x - 1, res_y - (j + 0.5) * res_y - 1, 1) = color_g * 255;
                //img(res_x - (i + 0.5) * res_x - 1, res_y - (j + 0.5) * res_y - 1, 2) = color_b * 255;

                image_mat[i1 * res_y * 3 + j1 * 3 + 0] = color_r * 255;
                image_mat[i1 * res_y * 3 + j1 * 3 + 1] = color_g * 255;
                image_mat[i1 * res_y * 3 + j1 * 3 + 2] = color_b * 255;
            }
        }
    }
}
#endif

#ifndef __CUDACC__
inline void raytrace_loop_cpu(int block_row, int block_col, tira::image<unsigned char>& img)
{
    int flag, k, u, min_k = 9999;
    float t_min, min_dist, shadow_r, shadow_g, shadow_b, color_r, color_g, color_b, shadow, t, dist, color;
    glm::vec3 plane_int, test, c, tri_int;
    double i, j;
    size_t i1, j1;
    ray pixel_ray;
    plane tri_p;
    tira::triangle tri;
    /*for (i = i_min - 0.5; i <= i_min - 0.4 ; i = i + 0.001)
        for (j = j_min - 0.5; j <= j_min - 0.4; j = j + 0.001)*/
    for (i1 = block_row; i1 < (block_row + 100) * res_x/1000; i1++)
        for (j1 = block_col; j1 < (block_col + 100) * res_y/1000; j1++)
        {
            //i = (i1 / res_x) - 0.5;
            //j = (j1 / res_y) - 0.5;
            i = (static_cast<double>(i1) / res_x) - 0.5;
            j = (static_cast<double>(j1) / res_y) - 0.5;
            flag = 0;
            t_min = 9999;
            shadow_r = 0;
            shadow_g = 0;
            shadow_b = 0;
            color_r = 0;
            color_g = 0;
            color_b = 0;
            shadow = 0;
            min_dist = 9999;
            pixel_ray.dir = cam.ray(i, j);
            pixel_ray.origin = cam.position();
            //test_intersect(pixel_ray, s, t);
            for (k = 0; k < num_sph; k++)
            {
                if (test_intersect(pixel_ray.origin, pixel_ray.dir, s[k], t))
                {
                    if (t > 0 and t < t_min)
                    {
                        c = lighting(pixel_ray, s, num_sph, k, t, l, num_lights);
                        t_min = t;
                        img(res_x - (i + 0.5) * res_x - 1, res_y - (j + 0.5) * res_y - 1, 0) = c.r;
                        img(res_x - (i + 0.5) * res_x - 1, res_y - (j + 0.5) * res_y - 1, 1) = c.g;
                        img(res_x - (i + 0.5) * res_x - 1, res_y - (j + 0.5) * res_y - 1, 2) = c.b;
                        flag = 1;
                    }

                    //img(i, j, 0) = s.r;
                    //img(i, j, 1) = s.g;
                    //img(i, j, 2) = s.b;
                }
            }
            if (flag == 0 && num_planes)       //if no sphere intersections, and the ray is not parallel to the plane
            {
                plane_int = plane_intersect(p, pixel_ray);

                /*if(int(i*10)==-4)
                    std::cout <<"\nhi"<< plane_int.x << plane_int.y << plane_int.z<<"\n";*/
                for (u = 0; u < num_lights; u++)
                {
                    //shadow = glm::dot(p.normal, pixel_ray.dir) / (glm::length(p.normal) * glm::length(pixel_ray.dir));
                    shadow = 1;
                    test = (plane_int - l[u].pos) / glm::length(plane_int - l[u].pos);
                    //std::cout << "\n" << test.x << test.y << test.z;
                    if (test_shadow(l[u].pos, test, s, num_sph, l[u], t))
                    {
                        shadow = 0;
                    }
                    else
                    {
                        //shadow = std::max(float(0),glm::dot(p.normal, -pixel_ray.dir) / (glm::length(p.normal) * glm::length(pixel_ray.dir)));
                        shadow = std::max(float(0), glm::dot(((l[u].pos - plane_int) / glm::length(l[u].pos - plane_int)), p.normal));
                        //std::cout <<"\n"<< shadow;
                    }
                    shadow_r = shadow_r + shadow * 255 * l[u].r;
                    shadow_g = shadow_g + shadow * 255 * l[u].g;
                    shadow_b = shadow_b + shadow * 255 * l[u].b;
                }
                img(res_x - (i + 0.5) * res_x - 1, res_y - (j + 0.5) * res_y - 1, 0) = p.r * shadow_r;
                img(res_x - (i + 0.5) * res_x - 1, res_y - (j + 0.5) * res_y - 1, 1) = p.g * shadow_g;
                img(res_x - (i + 0.5) * res_x - 1, res_y - (j + 0.5) * res_y - 1, 2) = p.b * shadow_b;
            }
            if (num_mesh && test_intersect(pixel_ray.origin, pixel_ray.dir, bound_sph, t))  //test intersect on bounding sphere
            {
                for (k = 0; k < mesh.count(); k++)
                {
                    tri = mesh[k];
                    tri_p.point = tri.v[0];
                    tri_p.normal = tri.n;
                    tri_int = plane_intersect(tri_p, pixel_ray);
                    if (test_triangle(tri, tri_int))
                    {
                        dist = glm::distance(tri_int, cam.position());
                        if (dist < min_dist)
                        {
                            for (u = 0; u < num_lights; u++)
                            {
                                if (u == 0)
                                {
                                    color_r = 0;
                                    color_g = 0;
                                    color_b = 0;
                                }
                                color = std::max(float(0), glm::dot(((l[u].pos - tri_int) / glm::length(l[u].pos - tri_int)), tri_p.normal));
                                color_r = glm::min(color_r + color * l[u].r, float(1));
                                color_g = glm::min(color_g + color * l[u].g, float(1));
                                color_b = glm::min(color_b + color * l[u].b, float(1));
                            }
                            min_k = k;
                            min_dist = dist;
                        }
                    }
                }
                if (min_k != -1)
                {
                    //color = 1;
                    img(res_x - (i + 0.5) * res_x - 1, res_y - (j + 0.5) * res_y - 1, 0) = color_r * 255;
                    img(res_x - (i + 0.5) * res_x - 1, res_y - (j + 0.5) * res_y - 1, 1) = color_g * 255;
                    img(res_x - (i + 0.5) * res_x - 1, res_y - (j + 0.5) * res_y - 1, 2) = color_b * 255;
                }
            }
        }
}
#endif

#endif